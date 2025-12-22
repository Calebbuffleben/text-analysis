"""
Servidor Socket.IO para comunicação em tempo real com backend NestJS.
Recebe chunks de transcrição e retorna resultados de análise.
"""

import socketio
import structlog
import base64
import time
from typing import Dict, Any
from .config import Config
from .types.messages import TranscriptionChunk, TextAnalysisResult, AudioChunk
from .services.analysis_service import TextAnalysisService
from .services.transcription_service import TranscriptionService
from .services.audio_buffer_service import AudioBufferService

logger = structlog.get_logger()

# Inicializar serviços
analysis_service = TextAnalysisService()
transcription_service = TranscriptionService()
audio_buffer_service = AudioBufferService(
    min_duration_sec=5.0,  # Agrupar pelo menos 5 segundos de áudio (alinhado com backend)
    max_duration_sec=10.0,  # Máximo 10 segundos antes de forçar transcrição
    flush_interval_sec=2.0  # Flush após 2 segundos sem novos chunks
)

# Configurar callback para quando buffer estiver pronto
async def on_buffer_ready(meeting_id: str, participant_id: str, track: str, 
                         wav_data: bytes, sample_rate: int, channels: int, timestamp: int):
    """Callback chamado quando buffer está pronto para transcrição"""
    try:
        logger.info(
            "🎙️ [BUFFER] Buffer pronto, iniciando transcrição",
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track,
            audio_size_bytes=len(wav_data)
        )
        
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
        
        # Enviar resultado via Socket.IO
        # Nota: Precisamos encontrar o sid correto para este participante
        # Por enquanto, enviaremos para todos os clientes conectados
        result_dict = result.model_dump()
        await sio.emit('text_analysis_result', result_dict)
        
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

audio_buffer_service.set_callback(on_buffer_ready)

# Criar servidor Socket.IO
# Configurações para melhor compatibilidade com Railway e polling HTTP
sio = socketio.AsyncServer(
    cors_allowed_origins=Config.SOCKETIO_CORS_ORIGINS,
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
)

# Handlers de conexão
@sio.event
async def connect(sid, environ):
    """Handler chamado quando cliente conecta"""
    logger.info(
        "✅ [SOCKET.IO] Cliente conectado",
        client_id=sid,
        remote_addr=environ.get('REMOTE_ADDR', 'unknown')
    )

@sio.event
async def disconnect(sid):
    """Handler chamado quando cliente desconecta"""
    logger.info(
        "❌ [SOCKET.IO] Cliente desconectado",
        client_id=sid
    )

# DIAGNÓSTICO: Registrar handler genérico para capturar todos os eventos
@sio.on('*')
async def catch_all(event, sid, data):
    """Handler genérico para capturar todos os eventos Socket.IO"""
    # Não logar eventos de sistema (connect, disconnect)
    if event not in ['connect', 'disconnect']:
        logger.info(
            "👂 [DIAGNÓSTICO] Evento Socket.IO recebido",
            event=event,
            client_id=sid,
            data_type=type(data).__name__,
            data_keys=list(data.keys()) if isinstance(data, dict) else 'not_dict'
        )
    print(f"[DIAGNÓSTICO] Evento genérico: {event}, sid={sid}, data={type(data)}")

# Criar app ASGI
app = socketio.ASGIApp(sio)

# Instanciar serviços (singletons)
logger.info("🔄 [SOCKET.IO] Inicializando serviços...")
analysis_service = TextAnalysisService()
transcription_service = TranscriptionService()
logger.info("✅ [SOCKET.IO] Serviços inicializados, Socket.IO server pronto")


@sio.event
async def connect(sid, environ):
    """
    Handler para conexão de cliente.
    
    Args:
        sid: Session ID do cliente
        environ: Informações do ambiente WSGI
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
    print(f"[DIAGNÓSTICO] audio_chunk chamado! sid={sid}, data_keys={list(data.keys()) if data else 'None'}")
    logger.info(
        "🔴 [DIAGNÓSTICO] Handler audio_chunk INICIADO",
        client_id=sid,
        data_type=type(data).__name__,
        data_keys=list(data.keys()) if isinstance(data, dict) else 'not_dict',
        has_meeting_id='meetingId' in data if isinstance(data, dict) else False
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
        
        # Buffer está pronto imediatamente! Processar via callback
        logger.info(
            "🎙️ [FLUXO] Buffer pronto imediatamente, processando via callback",
            client_id=sid,
            meeting_id=meeting_id,
            participant_id=participant_id,
            audio_size_bytes=len(combined_wav)
        )
        
        # Processar via callback (que faz transcrição + análise)
        await on_buffer_ready(
            meeting_id=meeting_id,
            participant_id=participant_id,
            track=track,
            wav_data=combined_wav,
            sample_rate=sample_rate,
            channels=channels,
            timestamp=timestamp
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
    await sio.emit('pong', {
        'timestamp': data.get('timestamp'),
        'service': 'text-analysis'
    }, room=sid)

