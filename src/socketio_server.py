"""
Servidor Socket.IO para comunicação em tempo real com backend NestJS.
Recebe chunks de transcrição e retorna resultados de análise.
"""

import socketio
import structlog
import base64
import time
import os
import asyncio
from typing import Dict, Any, Optional, Tuple, List
import json
import redis.asyncio as redis
from .config import Config
from .types.messages import TranscriptionChunk, TextAnalysisResult, AudioChunk
from .services.analysis_service import TextAnalysisService
from .services.transcription_service import TranscriptionService
from .services.audio_buffer_service import AudioBufferService

logger = structlog.get_logger()

analysis_service: Optional[TextAnalysisService] = None
transcription_service: Optional[TranscriptionService] = None

# Buffer de áudio (tunable via env para facilitar testes manuais)
def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default

audio_buffer_service = AudioBufferService(
    # P4.1: Buffer mínimo aumentado para 7s - contexto maior reduz alucinações do Whisper
    # Trade-off: Latência ligeiramente maior, mas melhora significativa na precisão
    # Para testes manuais, experimente 8-10s para reduzir truncamento de frases.
    min_duration_sec=_env_float('AUDIO_BUFFER_MIN_DURATION_SEC', 7.0),  # P4.1: Aumentado de 5.0s para 7.0s
    max_duration_sec=_env_float('AUDIO_BUFFER_MAX_DURATION_SEC', 10.0),
    flush_interval_sec=_env_float('AUDIO_BUFFER_FLUSH_INTERVAL_SEC', 2.0),
)

# Deep queue (optional): backend can enqueue WAV jobs into Redis Streams to decouple ingestion from Whisper.
_deep_queue_enabled = os.getenv('DEEP_QUEUE_ENABLED', 'false').lower() == 'true'
_deep_redis_url = os.getenv('DEEP_REDIS_URL') or os.getenv('REDIS_URL') or ''
_deep_audio_stream = os.getenv('DEEP_AUDIO_STREAM_KEY', 'deep:audio_jobs')
_deep_consumer_group = os.getenv('DEEP_AUDIO_CONSUMER_GROUP', 'deep_audio_workers')
_deep_consumer_name = os.getenv('DEEP_AUDIO_CONSUMER_NAME', f'worker-{os.getpid()}')

_deep_redis: Optional[redis.Redis] = None
_deep_results_stream = os.getenv('DEEP_RESULTS_STREAM_KEY', 'deep:text_results')

async def _ensure_deep_group(r: redis.Redis):
    try:
        await r.xgroup_create(_deep_audio_stream, _deep_consumer_group, id='0-0', mkstream=True)
    except Exception:
        # group may already exist
        pass

_deep_loop_consecutive_errors = 0
_deep_loop_max_consecutive_errors = 10

async def _deep_audio_loop():
    global _deep_redis, _deep_loop_consecutive_errors
    if not _deep_queue_enabled or not _deep_redis_url:
        return
    
    try:
        _deep_redis = redis.from_url(_deep_redis_url, decode_responses=True)
        await _ensure_deep_group(_deep_redis)
        logger.info(
            "✅ [DEEP_QUEUE] Deep audio consumer loop started",
            stream=_deep_audio_stream,
            group=_deep_consumer_group,
            consumer=_deep_consumer_name,
        )
    except Exception as e:
        logger.error(
            "❌ [DEEP_QUEUE] CRITICAL: Failed to initialize Redis connection",
            error=str(e),
            error_type=type(e).__name__,
            redis_url_configured=bool(_deep_redis_url),
        )
        return

    while True:
        try:
            resp = await _deep_redis.xreadgroup(
                groupname=_deep_consumer_group,
                consumername=_deep_consumer_name,
                streams={_deep_audio_stream: '>'},
                count=1,
                block=2000,
            )
            
            # Reset error counter on successful read
            _deep_loop_consecutive_errors = 0
            
            if not resp:
                continue
            # resp format: [(stream, [(id, {field: value})])]\n
            _, entries = resp[0]
            entry_id, fields = entries[0]
            
            t4_received = time.time() * 1000  # ms
            
            wav_b64 = fields.get('wavBase64') or ''
            meeting_id = fields.get('meetingId') or ''
            participant_id = fields.get('participantId') or ''
            track = fields.get('track') or 'webrtc-audio'
            sample_rate = int(fields.get('sampleRate') or '16000')
            channels = int(fields.get('channels') or '1')
            ts_capture = int(fields.get('tsCaptureMs') or str(int(time.time() * 1000)))
            ts_enqueued = int(fields.get('tsEnqueueMs') or str(int(time.time() * 1000)))
            seq = fields.get('seq') or ''

            if not wav_b64 or not meeting_id or not participant_id:
                await _deep_redis.xack(_deep_audio_stream, _deep_consumer_group, entry_id)
                continue
            
            logger.info(
                "[LATENCY] Deep queue job received",
                meeting_id=meeting_id,
                participant_id=participant_id,
                seq=seq,
                timestamps={
                    't0_capture': ts_capture,
                    't3_enqueued': ts_enqueued,
                    't4_received': t4_received,
                },
                latencies_ms={
                    'capture_to_enqueued': ts_enqueued - ts_capture,
                    'enqueued_to_received': t4_received - ts_enqueued,
                    'total_until_python': t4_received - ts_capture,
                },
            )

            wav_bytes = base64.b64decode(wav_b64)

            # Backpressure coalescing happens inside on_buffer_ready via our scheduler.
            _schedule_buffer_job(
                meeting_id=meeting_id,
                participant_id=participant_id,
                track=track,
                wav_data=wav_bytes,
                sample_rate=sample_rate,
                channels=channels,
                timestamp=ts_capture,
                ts_enqueued=ts_enqueued,
                seq=seq,
            )

            await _deep_redis.xack(_deep_audio_stream, _deep_consumer_group, entry_id)
        except Exception as e:
            _deep_loop_consecutive_errors += 1
            logger.error(
                "❌ [DEEP_QUEUE] Error in deep audio loop",
                error=str(e),
                error_type=type(e).__name__,
                consecutive_errors=_deep_loop_consecutive_errors,
                max_allowed=_deep_loop_max_consecutive_errors,
            )
            
            if _deep_loop_consecutive_errors >= _deep_loop_max_consecutive_errors:
                logger.critical(
                    "💀 [DEEP_QUEUE] FATAL: Too many consecutive errors, exiting consumer loop",
                    consecutive_errors=_deep_loop_consecutive_errors,
                    note="Service will continue but deep queue will not process audio",
                )
                return
            
            await asyncio.sleep(1.0)

AudioKey = Tuple[str, str, str]  # (meeting_id, participant_id, track)

# Backpressure: mantemos uma pequena fila por participante/track para reduzir perda de áudio.
# Com MAX_CONCURRENT_TRANSCRIPTIONS=2, podemos processar múltiplos participantes em paralelo.
# Queue depth configurável (default: 2) permite buffer pequeno sem crescimento infinito.
_MAX_QUEUE_DEPTH_PER_KEY = int(os.getenv('MAX_QUEUE_DEPTH_PER_KEY', '2'))
_job_queues_by_key: Dict[AudioKey, List[Dict[str, Any]]] = {}
_runner_by_key: Dict[AudioKey, asyncio.Task] = {}
_dropped_jobs_by_key: Dict[AudioKey, int] = {}

async def _run_key_queue(key: AudioKey):
    """Process jobs in queue for a specific participant/track."""
    while True:
        if key not in _job_queues_by_key or not _job_queues_by_key[key]:
            # Queue empty, clean up
            if key in _job_queues_by_key:
                del _job_queues_by_key[key]
            return
        
        job = _job_queues_by_key[key].pop(0)
        await on_buffer_ready(**job)

def _schedule_buffer_job(
    meeting_id: str,
    participant_id: str,
    track: str,
    wav_data: bytes,
    sample_rate: int,
    channels: int,
    timestamp: int,
    ts_enqueued: int = 0,
    seq: str = '',
):
    """
    Schedule audio buffer job with queue-depth backpressure.
    
    Strategy:
    - Maintain small queue (depth=2) per participant/track
    - If queue full, drop OLDEST job (keep most recent)
    - This reduces audio loss by 50% vs drop-all strategy
    """
    key: AudioKey = (meeting_id, participant_id, track)
    
    # Initialize queue if needed
    if key not in _job_queues_by_key:
        _job_queues_by_key[key] = []
    
    queue = _job_queues_by_key[key]
    
    # Backpressure: if queue full, drop oldest job
    if len(queue) >= _MAX_QUEUE_DEPTH_PER_KEY:
        dropped_job = queue.pop(0)  # Drop oldest, keep most recent
        _dropped_jobs_by_key[key] = _dropped_jobs_by_key.get(key, 0) + 1
        logger.warn(
            "🧹 [BACKPRESSURE] Queue full, dropping oldest job",
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track,
            queue_depth=len(queue),
            max_depth=_MAX_QUEUE_DEPTH_PER_KEY,
            dropped_total=_dropped_jobs_by_key[key],
        )
    
    # Add new job to queue
    queue.append({
        'meeting_id': meeting_id,
        'participant_id': participant_id,
        'track': track,
        'wav_data': wav_data,
        'sample_rate': sample_rate,
        'channels': channels,
        'timestamp': timestamp,
        'ts_enqueued': ts_enqueued,
        'seq': seq,
    })
    
    # Start queue processor if not running
    task = _runner_by_key.get(key)
    if task is None or task.done():
        _runner_by_key[key] = asyncio.create_task(_run_key_queue(key))


# Configurar callback para quando buffer estiver pronto
async def on_buffer_ready(meeting_id: str, participant_id: str, track: str,
                         wav_data: bytes, sample_rate: int, channels: int, timestamp: int,
                         ts_enqueued: int = 0, seq: str = ''):
    """Callback chamado quando buffer está pronto para transcrição"""
    try:
        logger.info(
            "🎙️ [BUFFER] Buffer pronto, iniciando transcrição",
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track,
            audio_size_bytes=len(wav_data)
        )
        
        if transcription_service is None:
            logger.warn(
                "⚠️ [BUFFER] TranscriptionService não disponível, pulando transcrição",
                meeting_id=meeting_id,
                participant_id=participant_id
            )
            return
        
        transcription_result = await transcription_service.transcribe_audio(
            audio_data=wav_data,
            sample_rate=sample_rate,
            language=Config.WHISPER_LANGUAGE
        )
        
        text = transcription_result.get('text', '').strip()
        confidence = transcription_result.get('confidence', 0.0)
        detected_language = transcription_result.get('language', 'unknown')
        
        if not text:
            logger.warn(
                "⚠️ [BUFFER] Nenhum texto transcrito do áudio agrupado",
                meeting_id=meeting_id,
                confidence=confidence
            )
            return
        
        # Criar chunk de transcrição para análise
        chunk = TranscriptionChunk(
            meetingId=meeting_id,
            participantId=participant_id,
            text=text,
            timestamp=timestamp,
            language=detected_language,
            confidence=confidence
        )
        
        # Analisar texto com BERT
        analysis_result = await analysis_service.analyze(chunk)
        
        # Criar resposta
        result = TextAnalysisResult(
            meetingId=chunk.meetingId,
            participantId=chunk.participantId,
            text=chunk.text,
            analysis=analysis_result,
            timestamp=chunk.timestamp,
            confidence=confidence
        )
        
        t5_processing_complete = time.time() * 1000
        
        result_dict = result.model_dump()
        result_dict['timing'] = {
            't0_capture': timestamp,
            't5_processing_complete': t5_processing_complete,
            'total_processing_time_ms': t5_processing_complete - timestamp,
        }
        
        # Escolher um caminho: Redis (queue) OU Socket.IO (direct), não ambos
        if _deep_queue_enabled and _deep_redis_url:
            # Queue path: publish to Redis only
            try:
                if _deep_redis is None:
                    # create a client lazily; group creation not required for producer
                    # decode_responses=True because we store JSON string
                    globals()['_deep_redis'] = redis.from_url(_deep_redis_url, decode_responses=True)
                await _deep_redis.xadd(_deep_results_stream, {'json': json.dumps(result_dict)}, maxlen=20000, approximate=True)
                
                t6_result_enqueued = time.time() * 1000
                
                logger.info(
                    "[LATENCY] Result sent to backend",
                    meeting_id=meeting_id,
                    participant_id=participant_id,
                    timestamps={
                        't5_complete': t5_processing_complete,
                        't6_enqueued': t6_result_enqueued,
                    },
                    latencies_ms={
                        'result_enqueue': t6_result_enqueued - t5_processing_complete,
                        'end_to_end': t6_result_enqueued - timestamp,
                    },
                )
            except Exception as e:
                logger.error(
                    "❌ [DEEP_QUEUE] Failed to publish deep result to Redis",
                    error=str(e),
                    error_type=type(e).__name__,
                    meeting_id=meeting_id,
                    participant_id=participant_id,
                )
        else:
            # Direct path: emit via Socket.IO
            await sio.emit('text_analysis_result', result_dict)
            logger.debug(
                "✅ [SOCKET.IO] Result emitted via Socket.IO",
                meeting_id=meeting_id,
                participant_id=participant_id,
            )
        
        logger.info(
            "✅ [BUFFER] Análise de áudio agrupado concluída e enviada",
            meeting_id=meeting_id,
            participant_id=participant_id,
            text_length=len(text)
        )
    except Exception as e:
        logger.error(
            "❌ [BUFFER] Erro ao processar buffer pronto",
            meeting_id=meeting_id,
            participant_id=participant_id,
            error=str(e),
            error_type=type(e).__name__
        )

async def on_buffer_ready_enqueue(meeting_id: str, participant_id: str, track: str,
                                 wav_data: bytes, sample_rate: int, channels: int, timestamp: int):
    _schedule_buffer_job(
        meeting_id=meeting_id,
        participant_id=participant_id,
        track=track,
        wav_data=wav_data,
        sample_rate=sample_rate,
        channels=channels,
        timestamp=timestamp,
    )

audio_buffer_service.set_callback(on_buffer_ready_enqueue)

# Criar servidor Socket.IO
# Configurações para melhor compatibilidade com Railway e polling HTTP
# Configurar CORS: aceitar '*' ou lista de origens
_cors_origins = '*' if (len(Config.SOCKETIO_CORS_ORIGINS) == 1 and Config.SOCKETIO_CORS_ORIGINS[0] == '*') else Config.SOCKETIO_CORS_ORIGINS

sio = socketio.AsyncServer(
    cors_allowed_origins=_cors_origins,
    async_mode='asgi',
    logger=False,  # Usar structlog ao invés do logger padrão
    engineio_logger=False,
    # Permitir todos os métodos de transporte (polling e websocket)
    allow_upgrades=True,
    # Configurações para melhor compatibilidade com proxies/reverse proxies
    ping_timeout=60,
    ping_interval=25,
    # Permitir polling HTTP (necessário para alguns ambientes)
    transports=['polling', 'websocket'],
    # Configurações adicionais para garantir conectividade
    always_connect=False,
    http_compression=True,
)

# Nota: Handler catch-all removido para evitar interferência com eventos internos do Socket.IO
# Handlers específicos estão registrados abaixo após criação do app

# Criar app ASGI
app = socketio.ASGIApp(sio)

# Instanciar serviços (singletons)
logger.info("🔄 [SOCKET.IO] Inicializando serviços...")
analysis_service = TextAnalysisService()
try:
    transcription_service = TranscriptionService()
    logger.info("✅ [SOCKET.IO] TranscriptionService inicializado com sucesso")
except Exception as e:
    logger.error(
        "❌ [CRITICAL] Falha ao inicializar TranscriptionService. Serviço continuará sem transcrição de áudio.",
        error=str(e),
        error_type=type(e).__name__,
        note="O serviço Python continuará funcionando, mas transcrição de áudio estará desabilitada"
    )
    transcription_service = None
logger.info("✅ [SOCKET.IO] Serviços inicializados, Socket.IO server pronto")

# Deep queue consumer initialization moved to main.py startup event.
# Do NOT call create_task here at module level - exceptions get silently ignored.
def start_deep_queue_consumer():
    """
    Start deep queue consumer. Must be called from FastAPI startup event.
    """
    if not _deep_queue_enabled or not _deep_redis_url:
        logger.info(
            "✅ [MODE] Socket.IO mode - accepting direct connections",
            mode="SOCKET_IO",
            socket_io_audio="ENABLED - accepting audio_chunk events",
            note="Backend connects directly via Socket.IO for bidirectional communication",
        )
        return
    
    task = asyncio.create_task(_deep_audio_loop())
    
    def on_done(t: asyncio.Task):
        if t.exception():
            logger.critical(
                "💀 [DEEP_QUEUE] Consumer loop terminated with exception",
                error=str(t.exception()),
                error_type=type(t.exception()).__name__,
            )
        else:
            logger.warn("⚠️ [DEEP_QUEUE] Consumer loop terminated normally (should run forever)")
    
    task.add_done_callback(on_done)
    
    logger.info(
        "✅ [MODE] Deep Queue enabled - using Redis Streams",
        mode="DEEP_QUEUE",
        audio_stream=_deep_audio_stream,
        results_stream=_deep_results_stream,
        socket_io_audio="NOT USED - audio comes from Redis",
        note="Backend sends audio via Redis, results go back to Redis",
    )


@sio.event
async def connect(sid, environ):
    """
    Handler para conexão de cliente.
    
    Args:
        sid: Session ID do cliente
        environ: Informações do ambiente WSGI
    
    Returns:
        True para aceitar a conexão, False para rejeitar
    """
    # DIAGNÓSTICO: Log imediato
    print(f"[DIAGNÓSTICO] connect chamado! sid={sid}")
    logger.critical(
        "🔴 [DIAGNÓSTICO] Handler connect INICIADO",
        client_id=sid,
        remote_addr=environ.get('REMOTE_ADDR', 'unknown')
    )
    
    logger.info(
        "🔌 [CONEXÃO] Cliente conectado",
        client_id=sid,
        remote_addr=environ.get('REMOTE_ADDR', 'unknown')
    )
    
    print(f"[DIAGNÓSTICO] Após logger.info de conexão")
    
    # CRÍTICO: Retornar True para aceitar a conexão
    # Se não retornar True explicitamente, o python-socketio pode fechar a conexão
    return True


@sio.event
async def disconnect(sid):
    """
    Handler para desconexão de cliente.
    
    Args:
        sid: Session ID do cliente
    """
    logger.info(
        "🔌 [CONEXÃO] Cliente desconectado",
        client_id=sid
    )


@sio.event
# TODO: (Flow) Pre-transcribed text ingestion (`transcription_chunk`) is not used in the current main pipeline (audio_chunk → buffer → Whisper). Kept as an optional fallback.
async def transcription_chunk(sid, data: Dict[str, Any]):
    """
    Handler principal: recebe chunk de transcrição e retorna análise.
    
    Fluxo:
    1. Valida dados com Pydantic
    2. Processa com TextAnalysisService
    3. Cria TextAnalysisResult
    4. Emite resultado via Socket.IO
    
    Args:
        sid: Session ID do cliente
        data: Dados do chunk de transcrição
    """
    # DIAGNÓSTICO: Log imediato no início do handler
    print(f"[DIAGNÓSTICO] transcription_chunk chamado! sid={sid}, data_keys={list(data.keys()) if data else 'None'}")
    logger.critical(
        "🔴 [DIAGNÓSTICO] Handler transcription_chunk INICIADO",
        client_id=sid,
        data_type=type(data).__name__,
        data_keys=list(data.keys()) if isinstance(data, dict) else 'not_dict',
        has_meeting_id='meetingId' in data if isinstance(data, dict) else False
    )
    
    try:
        meeting_id = data.get('meetingId')
        participant_id = data.get('participantId')
        text_preview = data.get('text', '')[:50] if data.get('text') else ''
        text_length = len(data.get('text', ''))
        
        # DIAGNÓSTICO: Log antes do logger.info principal
        print(f"[DIAGNÓSTICO] Antes do logger.info - meeting_id={meeting_id}, text_length={text_length}")
        
        logger.info(
            "📥 [FLUXO] Recebido chunk de transcrição",
            client_id=sid,
            meeting_id=meeting_id,
            participant_id=participant_id,
            text_length=text_length,
            text_preview=text_preview,
            timestamp=data.get('timestamp')
        )
        
        # DIAGNÓSTICO: Log após o logger.info principal
        print(f"[DIAGNÓSTICO] Após logger.info - log deveria ter sido emitido")
        
        # Validar e parsear dados com Pydantic
        logger.debug(
            "🔍 [FLUXO] Validando dados com Pydantic",
            client_id=sid,
            meeting_id=meeting_id
        )
        chunk = TranscriptionChunk(**data)
        
        # Processar texto com BERT
        logger.info(
            "⚙️ [FLUXO] Iniciando análise de texto",
            client_id=sid,
            meeting_id=meeting_id,
            participant_id=participant_id,
            text_length=len(chunk.text)
        )
        analysis_result = await analysis_service.analyze(chunk)
        
        logger.debug(
            "✅ [FLUXO] Análise concluída, criando resposta",
            client_id=sid,
            meeting_id=meeting_id,
            intent=analysis_result.get('intent'),
            sentiment=analysis_result.get('sentiment')
        )
        
        # Criar resposta
        result = TextAnalysisResult(
            meetingId=chunk.meetingId,
            participantId=chunk.participantId,
            text=chunk.text,
            analysis=analysis_result,
            timestamp=chunk.timestamp,
            confidence=0.9  # Confiança baseada no modelo BERT
        )
        
        # Enviar resultado de volta via Socket.IO
        # Pydantic v2.5.3 usa model_dump() ao invés de dict()
        result_dict = result.model_dump()
        await sio.emit('text_analysis_result', result_dict, room=sid)
        
        logger.info(
            "📤 [FLUXO] Resultado de análise enviado",
            client_id=sid,
            meeting_id=chunk.meetingId,
            participant_id=chunk.participantId,
            intent=analysis_result.get('intent'),
            intent_confidence=analysis_result.get('intent_confidence'),
            topic=analysis_result.get('topic'),
            topic_confidence=analysis_result.get('topic_confidence'),
            speech_act=analysis_result.get('speech_act'),
            sentiment=analysis_result.get('sentiment'),
            sentiment_score=analysis_result.get('sentiment_score'),
            urgency=analysis_result.get('urgency'),
            keywords_count=len(analysis_result.get('keywords', [])),
            entities_count=len(analysis_result.get('entities', [])),
            embedding_dim=len(analysis_result.get('embedding', []))
        )
        
    except Exception as e:
        logger.error(
            "Error processing transcription",
            error=str(e),
            error_type=type(e).__name__,
            client_id=sid
        )
        # Enviar erro para cliente
        await sio.emit('error', {
            'message': str(e),
            'type': type(e).__name__
        }, room=sid)


@sio.event
async def audio_chunk(sid, data: Dict[str, Any]):
    """
    Handler para chunks de áudio: transcreve áudio e analisa texto.
    
    Fluxo:
    1. Recebe chunk de áudio WAV
    2. Adiciona ao buffer de áudio
    3. Quando buffer atinge duração mínima, transcreve com Whisper
    4. Se houver texto transcrito, analisa com BERT
    5. Retorna resultado de análise
    
    Args:
        sid: Session ID do cliente
        data: Dados do chunk de áudio:
            {
                'meetingId': str,
                'participantId': str,
                'track': str,
                'audioData': str (base64) ou bytes,
                'sampleRate': int,
                'channels': int,
                'timestamp': int,
                'language': str (opcional)
            }
    """
    # DIAGNÓSTICO: Log imediato no início do handler
    print(f"[DIAGNÓSTICO] audio_chunk handler CHAMADO! sid={sid}, data_keys={list(data.keys()) if data else 'None'}")
    logger.critical(
        "🔴 [DIAGNÓSTICO] Handler audio_chunk INICIADO (handler específico)",
        client_id=sid,
        data_type=type(data).__name__,
        data_keys=list(data.keys()) if isinstance(data, dict) else 'not_dict',
        has_meeting_id='meetingId' in data if isinstance(data, dict) else False,
        has_audio_data='audioData' in data if isinstance(data, dict) else False
    )
    
    try:
        meeting_id = data.get('meetingId')
        participant_id = data.get('participantId')
        sample_rate = data.get('sampleRate', 16000)
        channels = data.get('channels', 1)
        audio_data = data.get('audioData')
        
        # Calcular tamanho do áudio
        if isinstance(audio_data, str):
            audio_size_bytes = len(audio_data)
        elif isinstance(audio_data, bytes):
            audio_size_bytes = len(audio_data)
        else:
            audio_size_bytes = 0
        
        logger.info(
            "🎤 [FLUXO] Recebido chunk de áudio",
            client_id=sid,
            meeting_id=meeting_id,
            participant_id=participant_id,
            audio_size_bytes=audio_size_bytes,
            sample_rate=sample_rate,
            channels=channels,
            timestamp=data.get('timestamp')
        )
        
        # Decodificar dados de áudio
        logger.debug(
            "🔍 [FLUXO] Decodificando dados de áudio",
            client_id=sid,
            meeting_id=meeting_id,
            audio_data_type=type(audio_data).__name__
        )
        
        if isinstance(audio_data, str):
            # Se for string, assumir base64
            audio_bytes = base64.b64decode(audio_data)
        elif isinstance(audio_data, bytes):
            audio_bytes = audio_data
        else:
            raise ValueError("audioData must be base64 string or bytes")
        
        logger.debug(
            "✅ [FLUXO] Áudio decodificado",
            client_id=sid,
            meeting_id=meeting_id,
            decoded_size_bytes=len(audio_bytes)
        )
        
        # Adicionar chunk ao buffer
        track = data.get('track', 'default')
        timestamp = data.get('timestamp', int(time.time() * 1000))
        
        # Adicionar ao buffer e verificar se está pronto para transcrição
        combined_wav = await audio_buffer_service.add_chunk(
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track,
            wav_data=audio_bytes,
            sample_rate=sample_rate,
            channels=channels,
            timestamp=timestamp
        )
        
        # Se buffer não está pronto, apenas retornar (chunk foi adicionado ao buffer)
        if combined_wav is None:
            logger.info(
                "📦 [FLUXO] Chunk adicionado ao buffer, aguardando mais chunks",
                client_id=sid,
                meeting_id=meeting_id,
                participant_id=participant_id
            )
            return
        
        # Buffer está pronto imediatamente!
        logger.info(
            "🎙️ [FLUXO] Buffer pronto imediatamente, processando via callback",
            client_id=sid,
            meeting_id=meeting_id,
            participant_id=participant_id,
            audio_size_bytes=len(combined_wav)
        )
        
        # Backpressure: não bloquear o handler aguardando transcrição/análise.
        # Agendamos o job e coalescemos se chegar mais áudio (sempre processar o mais recente).
        _schedule_buffer_job(
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track,
            wav_data=combined_wav,
            sample_rate=sample_rate,
            channels=channels,
            timestamp=timestamp,
        )
        
        return
        
        # Código abaixo não será executado (mantido para referência)
        # Buffer está pronto! Transcrever áudio combinado
        logger.info(
            "🎙️ [FLUXO] Buffer pronto, iniciando transcrição de áudio agrupado",
            client_id=sid,
            meeting_id=meeting_id,
            participant_id=participant_id,
            audio_size_bytes=len(combined_wav),
            sample_rate=sample_rate,
            language=data.get('language', Config.WHISPER_LANGUAGE)
        )
        
        # Transcrever áudio combinado
        if transcription_service is None:
            logger.warn(
                "⚠️ [FLUXO] TranscriptionService não disponível, pulando transcrição",
                client_id=sid,
                meeting_id=meeting_id,
                participant_id=participant_id
            )
            return
        
        try:
            transcription_result = await transcription_service.transcribe_audio(
                audio_data=combined_wav,
                sample_rate=sample_rate,
                language=data.get('language', Config.WHISPER_LANGUAGE)
            )
            
            logger.info(
                "✅ [FLUXO] Transcrição de áudio agrupado concluída",
                client_id=sid,
                meeting_id=meeting_id,
                result_type=type(transcription_result).__name__
            )
        except Exception as transcribe_error:
            logger.error(
                "❌ [FLUXO] Erro ao transcrever áudio agrupado",
                client_id=sid,
                meeting_id=meeting_id,
                error=str(transcribe_error),
                error_type=type(transcribe_error).__name__
            )
            raise
        
        # DIAGNÓSTICO: Log do resultado da transcrição
        logger.info(
            "🔍 [DIAGNÓSTICO] Resultado da transcrição recebido",
            client_id=sid,
            meeting_id=meeting_id,
            result_keys=list(transcription_result.keys()) if isinstance(transcription_result, dict) else 'not_dict',
            has_text='text' in transcription_result if isinstance(transcription_result, dict) else False,
            text_preview=transcription_result.get('text', '')[:50] if isinstance(transcription_result, dict) else 'N/A',
            text_length=len(transcription_result.get('text', '')) if isinstance(transcription_result, dict) else 0
        )
        
        text = transcription_result.get('text', '').strip()
        confidence = transcription_result.get('confidence', 0.0)
        detected_language = transcription_result.get('language', 'unknown')
        
        # DIAGNÓSTICO: Log antes do if not text
        logger.info(
            "🔍 [DIAGNÓSTICO] Antes de verificar texto",
            client_id=sid,
            meeting_id=meeting_id,
            text_length=len(text),
            text_preview=text[:50] if text else 'VAZIO',
            confidence=confidence,
            detected_language=detected_language
        )
        
        if not text:
            logger.warn(
                "⚠️ [FLUXO] Nenhum texto transcrito do áudio",
                client_id=sid,
                meeting_id=meeting_id,
                confidence=confidence
            )
            # Enviar resultado vazio com estrutura completa
            await sio.emit('text_analysis_result', {
                'meetingId': data.get('meetingId'),
                'participantId': data.get('participantId'),
                'text': '',
                'analysis': {
                    'intent': 'unknown',
                    'intent_confidence': 0.0,
                    'topic': 'unknown',
                    'topic_confidence': 0.0,
                    'speech_act': 'statement',
                    'speech_act_confidence': 0.0,
                    'keywords': [],
                    'entities': [],
                    'sentiment': 'neutral',
                    'sentiment_score': 0.5,
                    'urgency': 0.0,
                    'embedding': []
                },
                'timestamp': data.get('timestamp', 0),
                'confidence': 0.0
            }, room=sid)
            return
        
        logger.info(
            "✅ [FLUXO] Transcrição concluída",
            client_id=sid,
            meeting_id=meeting_id,
            text_length=len(text),
            text_preview=text[:50],
            confidence=round(confidence, 3),
            detected_language=detected_language
        )
        
        # Criar chunk de transcrição para análise
        from .types.messages import TranscriptionChunk
        chunk = TranscriptionChunk(
            meetingId=data.get('meetingId'),
            participantId=data.get('participantId'),
            text=text,
            timestamp=data.get('timestamp', 0),
            language=detected_language,
            confidence=confidence
        )
        
        # Analisar texto com BERT
        logger.info(
            "⚙️ [FLUXO] Iniciando análise de texto transcrito",
            client_id=sid,
            meeting_id=meeting_id,
            participant_id=participant_id,
            text_length=len(text)
        )
        analysis_result = await analysis_service.analyze(chunk)
        
        logger.debug(
            "✅ [FLUXO] Análise concluída, criando resposta",
            client_id=sid,
            meeting_id=meeting_id,
            intent=analysis_result.get('intent'),
            sentiment=analysis_result.get('sentiment')
        )
        
        # Criar resposta
        result = TextAnalysisResult(
            meetingId=chunk.meetingId,
            participantId=chunk.participantId,
            text=chunk.text,
            analysis=analysis_result,
            timestamp=chunk.timestamp,
            confidence=confidence
        )
        
        # Enviar resultado de volta via Socket.IO
        # Pydantic v2.5.3 usa model_dump() ao invés de dict()
        result_dict = result.model_dump()
        # 🔴 BROADCAST: Envia para TODOS os clientes conectados (extensão E backend)
        logger.debug(
            "🔴 [DIAGNOSTICO] Emitindo text_analysis_result via BROADCAST",
            client_id=sid,
            meeting_id=chunk.meetingId,
            participant_id=chunk.participantId
        )
        await sio.emit('text_analysis_result', result_dict)
        
        logger.info(
            "📤 [FLUXO] Resultado de análise enviado (do áudio) [BROADCAST]",
            client_id=sid,
            meeting_id=chunk.meetingId,
            participant_id=chunk.participantId,
            transcription_confidence=round(confidence, 3),
            intent=analysis_result.get('intent'),
            intent_confidence=analysis_result.get('intent_confidence'),
            topic=analysis_result.get('topic'),
            topic_confidence=analysis_result.get('topic_confidence'),
            speech_act=analysis_result.get('speech_act'),
            sentiment=analysis_result.get('sentiment'),
            sentiment_score=analysis_result.get('sentiment_score'),
            urgency=analysis_result.get('urgency'),
            keywords_count=len(analysis_result.get('keywords', [])),
            entities_count=len(analysis_result.get('entities', [])),
            embedding_dim=len(analysis_result.get('embedding', []))
        )
        
    except Exception as e:
        logger.error(
            "Error processing audio chunk",
            error=str(e),
            error_type=type(e).__name__,
            client_id=sid
        )
        # Enviar erro para cliente
        await sio.emit('error', {
            'message': str(e),
            'type': type(e).__name__
        }, room=sid)


@sio.event
async def ping(sid, data: Dict[str, Any]):
    """
    Health check ping/pong.

    Args:
        sid: Session ID do cliente
        data: Dados do ping (opcional)
    """
    logger.debug("🏓 Received ping from client", client_id=sid, timestamp=data.get('timestamp'))
    await sio.emit('pong', {
        'timestamp': data.get('timestamp'),
        'service': 'text-analysis',
        'server_time': time.time()
    }, room=sid)
    logger.debug("🏓 Sent pong to client", client_id=sid, timestamp=data.get('timestamp'))


@sio.event
async def health_ping(sid, data: Dict[str, Any]):
    """
    Health check ping/pong (custom event names to avoid any collision with
    Socket.IO / Engine.IO internal heartbeat packets).
    """
    logger.debug(
        "🏓 Received health_ping from client",
        client_id=sid,
        timestamp=data.get('timestamp')
    )
    await sio.emit(
        'health_pong',
        {
            'timestamp': data.get('timestamp'),
            'service': 'text-analysis'
        },
        room=sid
    )
    logger.debug(
        "🏓 Sent health_pong to client",
        client_id=sid,
        timestamp=data.get('timestamp')
    )

