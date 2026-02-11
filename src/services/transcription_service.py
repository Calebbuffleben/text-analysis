"""
Serviço de transcrição de áudio usando faster-whisper.
Recebe chunks de áudio WAV e retorna transcrições em texto.

faster-whisper é uma implementação otimizada do Whisper que:
- É mais rápida (até 4x mais rápida que openai-whisper)
- Usa menos memória
- Suporta os mesmos modelos (tiny, base, small, medium, large)
- Funciona melhor em CPU com compute_type="int8"
"""

import io
import asyncio
import time
import re
import structlog
import soundfile as sf
from threading import Lock
from faster_whisper import WhisperModel
import numpy as np
from scipy.ndimage import uniform_filter1d
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor
from ..config import Config

logger = structlog.get_logger()


class TranscriptionService:
    """
    Serviço de transcrição de áudio usando faster-whisper.
    
    faster-whisper é uma implementação otimizada do Whisper da OpenAI,
    otimizada para múltiplos idiomas incluindo português.
    
    Características:
    - Suporta múltiplos idiomas (português incluído)
    - Modelos leves disponíveis (tiny, base, small, medium, large)
    - Funciona em CPU e GPU
    - Mais rápido e eficiente que openai-whisper
    - Lazy loading do modelo (carrega apenas quando necessário)
    """
    
    def __init__(self):
        """
        Inicializa serviço de transcrição.
        O modelo Whisper pode ser pré-carregado no startup ou lazy-loaded na primeira transcrição.
        """
        self.model = None
        self._loaded = False
        self._load_lock = Lock()  # Thread-safe lock for model loading
        self._loading = False  # Flag to prevent concurrent loads
        self.model_name = Config.WHISPER_MODEL_NAME
        self.device = Config.WHISPER_DEVICE
        self.language = Config.WHISPER_LANGUAGE
        self.task = Config.WHISPER_TASK
        
        # faster-whisper compute_type: "int8" para CPU (mais rápido), "float16" para GPU
        # NÃO aceita "cpu" como valor - deve ser "int8" ou "float16"
        import os
        compute_type_env = os.getenv('WHISPER_COMPUTE_TYPE', '').strip().lower()
        if compute_type_env:
            # Validar e normalizar compute_type
            # Converter valores legados/comuns para valores válidos
            if compute_type_env == 'cpu':
                # Valor legado: converter "cpu" para "int8"
                self.compute_type = "int8"
                logger.warn(
                    "⚠️ [TRANSCRIÇÃO] WHISPER_COMPUTE_TYPE='cpu' é inválido, convertendo para 'int8'",
                    old_value="cpu",
                    new_value="int8",
                    note="faster-whisper requer 'int8' ou 'float16' para CPU, não 'cpu'"
                )
            elif compute_type_env in ('int8', 'int8_float16', 'float16'):
                # Valores válidos do faster-whisper
                self.compute_type = compute_type_env
            else:
                # Valor inválido: usar padrão baseado no device
                logger.warn(
                    "⚠️ [TRANSCRIÇÃO] WHISPER_COMPUTE_TYPE inválido, usando padrão baseado em device",
                    invalid_value=compute_type_env,
                    device=self.device,
                    note="Valores válidos: 'int8', 'int8_float16', 'float16'"
                )
                self.compute_type = "int8" if self.device == "cpu" else "float16"
        else:
            # Auto-detect: int8 para CPU, float16 para GPU
            self.compute_type = "int8" if self.device == "cpu" else "float16"
        
        # Log explícito do modelo que será usado
        env_value = os.getenv('WHISPER_MODEL_NAME', 'NOT_SET')
        logger.info(
            "🔍 [TRANSCRIÇÃO] Configuração do modelo faster-whisper",
            env_var_WHISPER_MODEL_NAME=env_value,
            config_WHISPER_MODEL_NAME=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
            note="faster-whisper é mais rápido que openai-whisper"
        )
        
        # Semáforo para limitar transcrições simultâneas
        # Configurável via MAX_CONCURRENT_TRANSCRIPTIONS (default: 2)
        # Permite processar múltiplos participantes em paralelo
        max_concurrent = Config.MAX_CONCURRENT_TRANSCRIPTIONS
        try:
            self._transcription_semaphore = asyncio.Semaphore(max_concurrent)
        except RuntimeError:
            # Se não houver event loop, criar None e inicializar depois
            self._transcription_semaphore = None
        self._max_concurrent = max_concurrent
        self._active_transcriptions = 0
        
        # Duração mínima de áudio para transcrição (em segundos)
        # Chunks muito pequenos (< 0.5s) são ignorados pois:
        # 1. faster-whisper funciona melhor com áudio mais longo
        # 2. Reduz carga desnecessária no CPU
        # 3. Melhora qualidade da transcrição
        self._min_audio_duration_sec = 0.5
        
        # Threshold RMS para detecção de fala (configurável via env)
        # Valores mais negativos = mais permissivo (detecta áudio mais baixo)
        # Valores mais altos = mais restritivo (requer áudio mais alto)
        # Padrão: -50dB (permissivo para ambientes de teste/Google Meet)
        rms_threshold_env = os.getenv('RMS_SPEECH_THRESHOLD_DB', '-50')
        try:
            self._rms_speech_threshold_db = float(rms_threshold_env)
        except ValueError:
            logger.warn(
                "⚠️ [TRANSCRIÇÃO] RMS_SPEECH_THRESHOLD_DB inválido, usando padrão",
                env_value=rms_threshold_env,
                default=-50.0
            )
            self._rms_speech_threshold_db = -50.0
        
        # ThreadPoolExecutor para executar faster-whisper em thread separada
        # faster-whisper é mais rápido mas ainda bloqueante, então executamos em thread separada
        # para não bloquear o event loop do asyncio
        # Número de workers = max_concurrent para permitir paralelismo
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)
        
        logger.info(
            "✅ [SERVIÇO] TranscriptionService inicializado",
            model=self.model_name,
            device=self.device,
            language=self.language,
            max_concurrent_transcriptions=max_concurrent,
            min_audio_duration_sec=self._min_audio_duration_sec,
            rms_speech_threshold_db=self._rms_speech_threshold_db
        )
    
    async def ensure_model_loaded(self):
        """
        Pre-load model during startup. Thread-safe and idempotent.
        Can be called multiple times safely - only loads once.
        
        This method uses double-check locking pattern to ensure:
        1. Only one thread loads the model at a time
        2. If model is already loaded, returns immediately
        3. If another thread is loading, waits for it to complete
        """
        # Fast path: already loaded
        if self._loaded:
            return
        
        # Acquire lock to check/set loading state
        with self._load_lock:
            # Double-check pattern: another thread might have loaded while waiting
            if self._loaded:
                return
            
            # Check if another thread is currently loading
            if self._loading:
                # Release lock and wait for loading to complete
                pass
            else:
                # This thread will load the model
                self._loading = True
        
        # If we set _loading to True, we're responsible for loading
        if self._loading and not self._loaded:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(self._executor, self._load_model)
                logger.info("✅ [TRANSCRIÇÃO] Model pre-loaded successfully during startup")
            finally:
                with self._load_lock:
                    self._loading = False
        else:
            # Another thread is loading, wait for it
            while self._loading and not self._loaded:
                await asyncio.sleep(0.1)
    
    def _load_model(self):
        """
        Carrega modelo faster-whisper.
        
        Modelos disponíveis (do menor ao maior):
        - tiny: ~39M parâmetros, mais rápido, menos preciso
        - base: ~74M parâmetros, bom equilíbrio
        - small: ~244M parâmetros, mais preciso
        - medium: ~769M parâmetros, muito preciso
        - large: ~1550M parâmetros, mais preciso, mais lento
        
        O modelo escolhido (base por padrão) oferece bom equilíbrio
        entre velocidade e precisão para transcrições em tempo real.
        
        faster-whisper é mais rápido que openai-whisper, especialmente em CPU
        com compute_type="int8".
        
        Thread-safe: Should only be called from ensure_model_loaded() which handles locking.
        """
        # Double-check: already loaded
        if self._loaded:
            return
        
        logger.info(
            "🔄 [TRANSCRIÇÃO] Carregando modelo faster-whisper",
            model=self.model_name,
            device=self.device,
            compute_type=self.compute_type
        )
        
        load_start = time.perf_counter()
        
        try:
            # faster-whisper compute_type deve ser "int8", "int8_float16" ou "float16"
            # device deve ser "cpu" ou "cuda"
            # Ele mesmo verifica se CUDA está disponível, então não precisamos verificar manualmente
            device = self.device
            compute_type = self.compute_type
            
            # Tentar carregar modelo faster-whisper
            # O modelo será baixado automaticamente na primeira execução
            # e armazenado em cache para uso futuro
            try:
                self.model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=compute_type
                )
            except (RuntimeError, ValueError) as cuda_error:
                # Se CUDA não estiver disponível ou houver erro, tentar com CPU
                if device == "cuda":
                    logger.warn(
                        "CUDA requested but not available, falling back to CPU",
                        error=str(cuda_error)
                    )
                    device = "cpu"
                    compute_type = "int8"
                    self.compute_type = "int8"
                    # Tentar novamente com CPU
                    self.model = WhisperModel(
                        self.model_name,
                        device=device,
                        compute_type=compute_type
                    )
                else:
                    # Re-raise se não for problema de CUDA
                    raise
            
            self._loaded = True
            load_latency_ms = (time.perf_counter() - load_start) * 1000
            
            logger.info(
                "✅ [TRANSCRIÇÃO] Modelo faster-whisper carregado com sucesso",
                model=self.model_name,
                device=device,
                compute_type=self.compute_type,
                language=self.language,
                load_time_ms=round(load_latency_ms, 2),
                note="faster-whisper é mais rápido que openai-whisper"
            )
            
        except Exception as e:
            load_latency_ms = (time.perf_counter() - load_start) * 1000
            logger.error(
                "❌ [TRANSCRIÇÃO] Falha ao carregar modelo faster-whisper",
                error=str(e),
                error_type=type(e).__name__,
                model=self.model_name,
                load_time_ms=round(load_latency_ms, 2)
            )
            raise
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcreve áudio WAV para texto usando Whisper.
        
        Como funciona:
        ==============
        1. O áudio WAV é decodificado para array numpy
        2. O Whisper processa o áudio em chunks sobrepostos
        3. O modelo gera tokens de texto correspondentes ao áudio
        4. Os tokens são decodificados para texto final
        
        Parâmetros:
        ===========
        - audio_data: Bytes do arquivo WAV (incluindo header)
        - sample_rate: Taxa de amostragem do áudio (Hz)
        - language: Idioma do áudio (None = auto-detect, 'pt' = português)
        
        Retorna:
        ========
        Dict com:
        {
            'text': str,              # Texto transcrito
            'language': str,           # Idioma detectado
            'segments': List[Dict],    # Segmentos com timestamps
            'confidence': float        # Confiança média (0-1)
        }
        
        Exemplo:
        ========
        result = service.transcribe_audio(wav_bytes, sample_rate=16000, language='pt')
        print(result['text'])  # "Olá, como você está?"
        """
        t_start = time.time() * 1000  # ms
        
        # Ensure model is loaded (thread-safe, idempotent)
        await self.ensure_model_loaded()
        
        # Inicializar semáforo se ainda não foi criado (fallback)
        if self._transcription_semaphore is None:
            self._transcription_semaphore = asyncio.Semaphore(self._max_concurrent)
        
        try:
            # FASE 2: Validações rápidas (<1ms, no event loop)
            # Validações que não bloqueiam o event loop
            if sample_rate <= 0:
                logger.warn(
                    "⚠️ [TRANSCRIÇÃO] Sample rate inválido",
                    sample_rate=sample_rate
                )
                return {
                    'text': '',
                    'language': language or self.language,
                    'segments': [],
                    'confidence': 0.0
                }
            
            # FASE 2: Pré-processamento no executor (não bloqueia event loop)
            # Todas as operações bloqueantes (decode_wav, RMS, trim, SNR) são executadas
            # em thread separada, permitindo que o event loop processe outros eventos
            # (ex: health_ping, outros handlers Socket.IO)
            
            # Obter event loop para executar no executor
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            
            logger.debug(
                "🔄 [TRANSCRIÇÃO] Iniciando pré-processamento em thread separada (não bloqueia event loop)",
                audio_size_bytes=len(audio_data),
                expected_sample_rate=sample_rate
            )
            
            # Executar pré-processamento no executor (FASE 2)
            # Operações bloqueantes: _decode_wav(), _has_speech_rms(), _trim_silence(), _estimate_snr()
            # Isso libera o event loop para processar outros eventos (ex: health_ping)
            preprocess_result = await loop.run_in_executor(
                self._executor,
                self._preprocess_audio_sync,
                audio_data,
                sample_rate,
                language
            )
            
            t_after_preprocess = time.time() * 1000  # ms
            
            # NOVO: Verificar se áudio foi classificado como tendo fala
            # Se has_speech=False, retornar imediatamente SEM processamento Whisper
            if not preprocess_result.get('has_speech', False):
                reason = preprocess_result.get('rejection_reason', 'unknown')
                rms_db = preprocess_result.get('rms_max_db', -float('inf'))
                audio_duration = preprocess_result.get('audio_duration_sec', 0.0)
                
                # Log de latência para áudio rejeitado rapidamente
                logger.info(
                    "[LATENCY] Audio rejected quickly",
                    latencies_ms={
                        'preprocess_only': round(t_after_preprocess - t_start, 2),
                        'whisper_skipped': True,
                        'total': round(t_after_preprocess - t_start, 2)
                    },
                    classification={
                        'has_speech': False,
                        'reason': reason,
                        'rms_max_db': round(rms_db, 2),
                        'rms_threshold_db': self._rms_speech_threshold_db,
                        'audio_duration_sec': round(audio_duration, 3)
                    },
                    audio_size_bytes=len(audio_data),
                    note="Áudio descartado SEM custo de transcrição Whisper - pipeline continua fluindo"
                )
                
                return {
                    'text': '',
                    'language': language or self.language,
                    'segments': [],
                    'confidence': 0.0,
                    'has_speech': False,
                    'rejection_reason': reason,
                    'rms_max_db': rms_db
                }
            
            # Se chegou aqui: has_speech=True, extrair dados do pré-processamento para transcrição
            # Dado do áudio decodificado e processado para transcrição            
            audio_array = preprocess_result['audio_array']
            # Informações do trim de silêncio
            trim_info = preprocess_result['trim_info']
            # SNR estimado
            estimated_snr_db = preprocess_result['estimated_snr_db']
            # Tamanho do beam search
            beam_size = preprocess_result['beam_size']
            # Se o VAD foi usado
            use_vad = preprocess_result['use_vad']
            # Duração do áudio após o trim de silêncio
            audio_duration_sec_after_trim = preprocess_result['audio_duration_sec_after_trim']
            # Opções de transcrição
            transcribe_options = preprocess_result['transcribe_options']
            
            logger.debug(
                "✅ [TRANSCRIÇÃO] Pré-processamento concluído",
                audio_samples=len(audio_array),
                audio_length_sec=round(audio_duration_sec_after_trim, 2),
                trim_info=trim_info,
                estimated_snr_db=round(estimated_snr_db, 2),
                beam_size=beam_size,
                vad_filter=use_vad
            )
            
            logger.info(
                "🎙️ [TRANSCRIÇÃO] Iniciando transcrição com Whisper",
                audio_length_sec=round(audio_duration_sec_after_trim, 2),
                audio_samples=len(audio_array),
                sample_rate=preprocess_result['sample_rate'],
                language=transcribe_options['language'],
                model=self.model_name,
                device=self.device
            )
            
            # Transcrição com Whisper (semáforo + executor)
            # O semáforo é usado APENAS para limitar transcrições simultâneas do Whisper
            # Pré-processamento já foi executado no executor, fora do semáforo
            # Isso garante que o event loop permanece livre durante pré-processamento
            
            # Inicializar variáveis de latência antes de adquirir semáforo
            transcribe_start = time.perf_counter()
            transcribe_latency_ms = 0.0
            
            logger.debug(
                "🔄 [TRANSCRIÇÃO] Aguardando slot disponível para transcrição",
                audio_length_sec=round(audio_duration_sec_after_trim, 2),
                active_transcriptions=self._active_transcriptions
            )
            
            # Adquirir semáforo APENAS para transcrição do Whisper
            # Pré-processamento não precisa do semáforo (já executado no executor)
            # Semáforo garante que apenas uma transcrição Whisper execute por vez
            
            async with self._transcription_semaphore:
                self._active_transcriptions += 1
                result = None
                
                try:
                    # Verificar se modelo está carregado
                    if self.model is None:
                        logger.error("❌ [TRANSCRIÇÃO] Modelo faster-whisper não está carregado!")
                        transcribe_latency_ms = (time.perf_counter() - transcribe_start) * 1000
                        return {
                            'text': '',
                            'language': language or self.language,
                            'segments': [],
                            'confidence': 0.0
                        }
                    
                    logger.info(
                        "⏳ [TRANSCRIÇÃO] Chamando faster-whisper model.transcribe",
                        active_transcriptions=self._active_transcriptions,
                        audio_samples=len(audio_array),
                        audio_length_sec=round(audio_duration_sec_after_trim, 2),
                        model=self.model_name,
                        compute_type=self.compute_type,
                        timeout_sec=30.0
                    )
                    
                    # Criar função de transcrição para o executor
                    # faster-whisper retorna (segments, info) ao invés de dict
                    # Referenciar model, audio, options, language e sample_rate no closure
                    model_ref = self.model
                    # Copiar audio_array para evitar modificações no original
                    audio_ref = audio_array.copy()
                    # Copiar options para evitar modificações no original
                    options_ref = transcribe_options.copy()
                    # Capturar language no closure
                    language_ref = language or self.language 
                    # Capturar sample_rate no closure
                    sample_rate_ref = preprocess_result['sample_rate']  
                    # Criar função de transcrição para o executor
                    def transcribe_sync():
                        try:
                            # faster-whisper retorna (segments, info)
                            # Transcrição com Whisper (semáforo + executor)
                            # segments é um iterador de objetos Segment
                            segments, info = model_ref.transcribe(audio_ref, **options_ref)
                            
                            # Converter segments para lista e processar
                            segments_list = list(segments)
                            
                            # Construir texto completo concatenando todos os segmentos
                            text = " ".join(seg.text for seg in segments_list).strip()
                            
                            # Aplicar pós-processamento para corrigir erros comuns de transcrição
                            text = self._fix_common_transcription_errors(text)
                            
                            # Converter segments para formato dict compatível
                            segments_dict = []
                            for seg in segments_list:
                                segments_dict.append({
                                    'start': seg.start,
                                    'end': seg.end,
                                    'text': seg.text,
                                    'no_speech_prob': getattr(seg, 'no_speech_prob', 0.0),
                                    'compression_ratio': getattr(seg, 'compression_ratio', 0.0),
                                    'avg_logprob': getattr(seg, 'avg_logprob', 0.0),
                                })
                            
                            # Retornar formato compatível com openai-whisper
                            return {
                                'text': text,
                                'language': info.language if hasattr(info, 'language') else language_ref,
                                'language_probability': getattr(info, 'language_probability', 1.0),
                                'segments': segments_dict,
                                '_raw_segments_count': len(segments_list),
                                'duration': getattr(info, 'duration', len(audio_ref) / sample_rate_ref)
                            }
                        except Exception as e:
                            logger.error(f"Erro dentro do transcribe_sync: {e}")
                            raise
                    
                    # Adicionar timeout de 30 segundos
                    # faster-whisper é mais rápido: tiny < 2s, base < 5s, small < 10s para 8s de áudio
                    task = loop.run_in_executor(self._executor, transcribe_sync)
                    result = await asyncio.wait_for(task, timeout=30.0)
                    transcribe_latency_ms = (time.perf_counter() - transcribe_start) * 1000
                    
                    logger.info(
                        "✅ [TRANSCRIÇÃO] faster-whisper retornou resultado",
                        latency_ms=round(transcribe_latency_ms, 2),
                        result_type=type(result).__name__,
                        has_text='text' in result if isinstance(result, dict) else False
                    )
                except asyncio.TimeoutError:
                    transcribe_latency_ms = (time.perf_counter() - transcribe_start) * 1000
                    logger.error(
                        "⏱️ [TRANSCRIÇÃO] Timeout na transcrição (30s excedido)",
                        latency_ms=round(transcribe_latency_ms, 2),
                        audio_length_sec=round(audio_duration_sec_after_trim, 2),
                        model=self.model_name
                    )
                    result = None
                except Exception as executor_error:
                    transcribe_latency_ms = (time.perf_counter() - transcribe_start) * 1000
                    logger.error(
                        "❌ [TRANSCRIÇÃO] Erro no executor do faster-whisper",
                        error=str(executor_error),
                        error_type=type(executor_error).__name__,
                        latency_ms=round(transcribe_latency_ms, 2)
                    )
                    result = None
                finally:
                    # Sempre decrementar contador, mesmo em caso de erro
                    self._active_transcriptions -= 1
                    logger.debug(
                        "🔓 [TRANSCRIÇÃO] Semáforo liberado",
                        active_transcriptions=self._active_transcriptions
                    )
            
            # Verificar se result foi definido
            if result is None:
                logger.warn(
                    "⚠️ [TRANSCRIÇÃO] Resultado da transcrição é None",
                    audio_length_sec=round(audio_duration_sec_after_trim, 2),
                    reason="transcribe_sync() retornou None - possible causes: exception during transcription, timeout, or model failure"
                )
                return {
                    'text': '',
                    'language': language or self.language,
                    'segments': [],
                    'confidence': 0.0
                }
            
            # Extrair informações relevantes
            text_raw = result.get('text', '')
            text = text_raw.strip()
            detected_language = result.get('language', language or self.language)
            segments = result.get('segments', [])
            raw_segments_count = int(result.get('_raw_segments_count', len(segments) if isinstance(segments, list) else 0))
            used_unfiltered_segments = bool(result.get('_used_unfiltered_segments', False))
            
            # Log detalhado quando texto é extraído mas está vazio
            if not text and text_raw:
                logger.warn(
                    "⚠️ [TRANSCRIÇÃO] Texto extraído está vazio após strip()",
                    text_raw_length=len(text_raw),
                    text_raw_preview=text_raw[:50],
                    segments_count=len(segments),
                    reason="Whisper retornou texto não-vazio mas strip() resultou em string vazia (apenas espaços/tabs/newlines)"
                )
            elif not text:
                logger.warn(
                    "⚠️ [TRANSCRIÇÃO] Texto extraído está vazio (text_raw também vazio)",
                    segments_count=len(segments),
                    reason="Whisper não detectou nenhum texto transcrito - possible causes: silence only, audio too short, low quality"
                )

                # Retry automático: se o Whisper entregou zero segmentos/texto vazio, repetir com
                # VAD desligado e thresholds mais permissivos, porque isso acontece frequentemente
                # no fim de conversas longas (ruído + sobreposição de vozes).
                retry_options = transcribe_options.copy()
                retry_options.update({
                    'vad_filter': False,
                    'no_speech_threshold': 0.8,
                    'log_prob_threshold': -2.0,
                })

                logger.warn(
                    "🔁 [TRANSCRIÇÃO] Retry por texto vazio/0 segmentos (modo permissivo)",
                    original_segments_count=len(segments),
                    original_raw_segments_count=raw_segments_count,
                    original_used_unfiltered_segments=used_unfiltered_segments,
                    audio_length_sec=round(audio_duration_sec_after_trim, 2),
                )

                async with self._transcription_semaphore:
                    self._active_transcriptions += 1
                    try:
                        model_ref = self.model
                        audio_ref = audio_array.copy()
                        options_ref = retry_options.copy()
                        language_ref = language or self.language
                        sample_rate_ref = preprocess_result['sample_rate']

                        def transcribe_sync_retry():
                            segments, info = model_ref.transcribe(audio_ref, **options_ref)
                            segments_list = list(segments)
                            # No retry, não aplicar filtro P3.1 (é justamente o que pode zerar tudo)
                            text_retry = " ".join(seg.text for seg in segments_list).strip()
                            text_retry = self._fix_common_transcription_errors(text_retry)
                            segments_dict_retry = [{
                                'start': seg.start,
                                'end': seg.end,
                                'text': seg.text,
                                'no_speech_prob': getattr(seg, 'no_speech_prob', 0.0),
                                'compression_ratio': getattr(seg, 'compression_ratio', 0.0),
                                'avg_logprob': getattr(seg, 'avg_logprob', 0.0),
                            } for seg in segments_list]
                            return {
                                'text': text_retry,
                                'language': info.language if hasattr(info, 'language') else language_ref,
                                'segments': segments_dict_retry,
                                '_raw_segments_count': len(segments_list),
                                '_used_unfiltered_segments': True,
                            }

                        task = loop.run_in_executor(self._executor, transcribe_sync_retry)
                        retry_result = await asyncio.wait_for(task, timeout=30.0)
                        # Sobrescrever o resultado se o retry trouxe texto
                        retry_text = (retry_result.get('text', '') or '').strip()
                        retry_segments = retry_result.get('segments', []) or []
                        if retry_text:
                            result = retry_result
                            text_raw = result.get('text', '')
                            text = text_raw.strip()
                            detected_language = result.get('language', detected_language)
                            segments = retry_segments
                            raw_segments_count = int(result.get('_raw_segments_count', len(segments)))
                            used_unfiltered_segments = bool(result.get('_used_unfiltered_segments', True))
                            logger.info(
                                "✅ [TRANSCRIÇÃO] Retry recuperou texto",
                                text_length=len(text),
                                segments_count=len(segments),
                                raw_segments_count=raw_segments_count,
                            )
                    except Exception as retry_error:
                        logger.warn(
                            "⚠️ [TRANSCRIÇÃO] Retry falhou",
                            error=str(retry_error),
                            error_type=type(retry_error).__name__,
                        )
                    finally:
                        self._active_transcriptions -= 1
            
            t_after_transcription = time.time() * 1000  # ms
            
            logger.info(
                "[LATENCY] Transcription timing",
                latencies_ms={
                    'preprocess': round(t_after_preprocess - t_start, 2),
                    'transcription': round(t_after_transcription - t_after_preprocess, 2),
                    'total': round(t_after_transcription - t_start, 2),
                    'whisper_skipped': False
                },
                classification={
                    'has_speech': True,
                    'rms_max_db': round(preprocess_result.get('rms_max_db', 0.0), 2),
                    'rms_threshold_db': self._rms_speech_threshold_db,
                    'audio_duration_sec_after_trim': round(preprocess_result.get('audio_duration_sec_after_trim', 0.0), 3)
                },
                result={
                    'text_length': len(text) if 'text' in locals() and text else 0,
                    'confidence': result.get('confidence', 0.0) if result else 0.0,
                    'segments_count': len(segments) if 'segments' in locals() and segments else 0
                },
                note="Áudio com fala processado completamente via Whisper"
            )
            
            # Calcular confiança média dos segmentos (após filtro P3.1)
            confidence = 0.0
            if segments:
                confidences = [
                    seg.get('no_speech_prob', 0.0) for seg in segments
                    if 'no_speech_prob' in seg
                ]
                if confidences:
                    # no_speech_prob é a probabilidade de NÃO ter fala
                    # Queremos a probabilidade de TER fala, então: 1 - no_speech_prob
                    speech_probs = [1.0 - conf for conf in confidences]
                    confidence = float(np.mean(speech_probs)) if speech_probs else 0.0
            
            # Detectar repetições no texto (problema comum com áudio ruim)
            text_words = text.split()
            unique_words = set(text_words)
            repetition_ratio = 1.0 - (len(unique_words) / len(text_words)) if text_words else 0.0
            has_repetition = repetition_ratio > 0.3  # Mais de 30% de repetição
            
            # P3.2: Validar texto agregado antes de retornar
            # REVISADO: Reduzir min_confidence de 0.3 para 0.15 para permitir transcrições válidas com baixa confiança
            # Repetition ratio mantido em 0.5 para evitar alucinações repetitivas
            min_confidence = 0.15  # Confidence mínima aceitável (reduzido de 0.3 para 0.15)
            max_repetition_ratio = 0.5  # Repetition ratio máximo aceitável (50%)
            
            # Log detalhado antes da validação
            logger.debug(
                "🔍 [TRANSCRIÇÃO] Validando qualidade do texto transcrito",
                text_length=len(text),
                text_preview=text[:50] if text else '',
                confidence=round(confidence, 3),
                min_confidence=min_confidence,
                repetition_ratio=round(repetition_ratio, 3),
                max_repetition_ratio=max_repetition_ratio,
                segments_count=len(segments),
                will_pass_confidence=confidence >= min_confidence,
                will_pass_repetition=repetition_ratio <= max_repetition_ratio
            )
            
            if confidence < min_confidence or repetition_ratio > max_repetition_ratio:
                # Não descartar texto útil que contém sinais claros de indecisão.
                # Isso é crucial no fim de conversas longas, onde a qualidade do áudio tende a piorar.
                text_lower = text.lower() if text else ''
                indecision_lexicon = [
                    'adiar', 'depois', 'talvez', 'preciso pensar', 'vamos ver',
                    'por enquanto', 'não sei', 'ainda não', 'mais tarde', 'pensar melhor',
                ]
                has_indecision_lexicon = any(tok in text_lower for tok in indecision_lexicon)
                if text and repetition_ratio <= max_repetition_ratio and has_indecision_lexicon:
                    logger.warn(
                        "⚠️ [TRANSCRIÇÃO] Mantendo texto apesar de baixa confiança (léxico de indecisão)",
                        confidence=round(confidence, 3),
                        min_confidence=min_confidence,
                        text_preview=text[:100],
                        segments_count=len(segments),
                    )
                    return {
                        'text': text,
                        'language': detected_language,
                        'segments': segments,
                        'confidence': confidence
                    }

                logger.warn(
                    "⚠️ [TRANSCRIÇÃO] Texto rejeitado por baixa qualidade",
                    confidence=round(confidence, 3),
                    min_confidence=min_confidence,
                    confidence_gap=round(min_confidence - confidence, 3) if confidence < min_confidence else 0,
                    repetition_ratio=round(repetition_ratio, 3),
                    max_repetition_ratio=max_repetition_ratio,
                    repetition_excess=round(repetition_ratio - max_repetition_ratio, 3) if repetition_ratio > max_repetition_ratio else 0,
                    text_preview=text[:100] if text else '',
                    text_full=text if len(text) <= 200 else text[:200] + '...',
                    segments_count=len(segments),
                    reason='low_confidence' if confidence < min_confidence else 'high_repetition',
                    note="min_confidence reduzido de 0.3 para 0.15 para permitir transcrições válidas com baixa confiança"
                )
                return {
                    'text': '',  # Retornar texto vazio em vez de alucinações
                    'language': detected_language,
                    'segments': [],  # Também retornar segments vazio
                    'confidence': 0.0
                }
            
            # Log detalhado dos segmentos para diagnóstico
            segment_previews = []
            if segments:
                for i, seg in enumerate(segments[:3]):  # Primeiros 3 segmentos
                    seg_text = seg.get('text', '').strip()
                    seg_no_speech = seg.get('no_speech_prob', 0.0)
                    segment_previews.append({
                        'index': i,
                        'text_preview': seg_text[:30] if seg_text else '',
                        'no_speech_prob': round(seg_no_speech, 2),
                        'start': round(seg.get('start', 0), 2),
                        'end': round(seg.get('end', 0), 2)
                    })
            
            logger.info(
                "✅ [TRANSCRIÇÃO] Transcrição concluída",
                text_length=len(text),
                text_preview=text[:100] if text else '',  # Aumentar preview para 100 chars
                text_full=text if len(text) <= 200 else text[:200] + '...',  # Texto completo se curto
                language=detected_language,
                confidence=round(confidence, 3),
                segments_count=len(segments),
                repetition_ratio=round(repetition_ratio, 2),
                has_repetition=has_repetition,
                segment_previews=segment_previews,
                latency_ms=round(transcribe_latency_ms, 2),
                warning="Repetição detectada" if has_repetition else None
            )
            
            return {
                'text': text,
                'language': detected_language,
                'segments': segments,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(
                "Transcription failed",
                error=str(e),
                error_type=type(e).__name__
            )
            # Retornar resultado vazio em caso de erro
            return {
                'text': '',
                'language': language or self.language,
                'segments': [],
                'confidence': 0.0
            }
    
    def _decode_wav(self, wav_data: bytes, expected_sample_rate: int) -> Optional[np.ndarray]:
        """
        Decodifica dados WAV para array numpy usando soundfile.
        O soundfile lida automaticamente com a decodificação e normalização.     
        Args:
            wav_data: Bytes do arquivo WAV (incluindo header)
            expected_sample_rate: Taxa de amostragem desejada (Hz)
        Returns:
            Array numpy float32 normalizado (-1.0 a 1.0), mono, na taxa expected_sample_rate.
            None em caso de erro na decodificação.
        """
        try:
            # Ler WAV usando soundfile (suporta PCM 16/24/32-bit e float nativamente)
            # soundfile já retorna float32 normalizado em [-1, 1]
            audio_float, sample_rate = sf.read(io.BytesIO(wav_data), dtype='float32')
            
            # Converter estéreo para mono (média dos canais)
            if audio_float.ndim == 2:
                audio_float = audio_float.mean(axis=1)
            
            # Resample se necessário (Whisper funciona melhor com 16kHz)
            # Nota: Se o áudio já estiver na taxa esperada, não precisa resample
            if sample_rate != expected_sample_rate:
                try:
                    from scipy import signal
                    original_samples = len(audio_float)
                    num_samples = int(original_samples * expected_sample_rate / sample_rate)
                    if num_samples > 0:
                        audio_float = signal.resample(audio_float, num_samples)
                        logger.debug(
                            "Audio resampled",
                            from_rate=sample_rate,
                            to_rate=expected_sample_rate,
                            original_samples=original_samples,
                            resampled_samples=num_samples
                        )
                    else:
                        logger.warn("Invalid resample target, keeping original sample rate")
                except ImportError:
                    logger.warn("scipy not available, skipping resample - Whisper will handle it")
                    # Whisper pode lidar com diferentes sample rates, mas 16kHz é ideal
                except Exception as e:
                    logger.warn(f"Resample failed: {e}, keeping original sample rate")
            else:
                logger.debug("Audio already at target sample rate, no resample needed")
            
            return audio_float
                
        except Exception as e:
            logger.error("Failed to decode WAV", error=str(e))
            return None
    
    def _rms_max_over_windows(self, audio_array: np.ndarray, window_samples: int) -> float:
        """
        Usa média móvel (uniform_filter1d) sobre o sinal ao quadrado para obter
        RMS por "frame" e retorna o máximo.
        
        Args:
            audio_array: Array numpy de áudio normalizado (-1 a 1)
            window_samples: Tamanho da janela em amostras (>= 1)
        
        Returns:
            Máximo RMS encontrado em qualquer janela (linear, não dB).
            Retorna 0.0 se audio_array estiver vazio.
        """
        if len(audio_array) == 0:
            return 0.0
        if window_samples < 1:
            window_samples = 1
        size = min(window_samples, len(audio_array))
        squared = audio_array.astype(np.float64) ** 2
        mean_sq = uniform_filter1d(squared, size=size, mode='constant', cval=0.0)
        rms_per_frame = np.sqrt(np.maximum(mean_sq, 0.0))
        return float(np.max(rms_per_frame))
    
    def _has_speech_rms(self, audio_array: np.ndarray, sample_rate: int, 
                       rms_threshold_db: float = -40.0, window_size_ms: int = 100) -> bool:
        """
        P1.2: Detecta se o áudio contém fala baseado em RMS (Root Mean Square).
        
        Calcula RMS em janelas curtas e verifica se algum trecho tem energia suficiente
        para indicar fala. Mais leve que VAD completo e eficaz para filtrar silêncio.
        
        Args:
            audio_array: Array numpy de áudio normalizado (-1 a 1)
            sample_rate: Taxa de amostragem em Hz
            rms_threshold_db: Threshold em dB (padrão -40dB, mais alto = mais restritivo)
            window_size_ms: Tamanho da janela em ms para calcular RMS
        
        Returns:
            True se detectar fala, False se apenas silêncio/ruído
        """
        if len(audio_array) == 0:
            return False
        
        # Validar sample_rate antes de usar
        if sample_rate <= 0:
            logger.warn("⚠️ [PRÉ-PROCESSAMENTO] Sample rate inválido para detecção RMS", sample_rate=sample_rate)
            return False
        
        # Converter threshold dB para linear (RMS)
        # dB = 20 * log10(RMS)
        # RMS = 10^(dB/20)
        rms_threshold_linear = 10 ** (rms_threshold_db / 20.0)
        
        # Calcular tamanho da janela em samples
        window_samples = int(sample_rate * window_size_ms / 1000.0)
        if window_samples < 1:
            window_samples = 1
        
        max_rms = self._rms_max_over_windows(audio_array, window_samples)
        has_speech = max_rms >= rms_threshold_linear
        
        logger.info(
            "🔍 [RMS_DEBUG] Verificação RMS de fala",
            max_rms_db=round(20 * np.log10(max_rms + 1e-10), 2),
            threshold_db=rms_threshold_db,
            has_speech=has_speech,
            window_samples=window_samples
        )
        
        return has_speech
    
    def _has_speech_rms_with_level(self, audio_array: np.ndarray, sample_rate: int) -> tuple:
        """
        Detecta fala via RMS e retorna o nível máximo encontrado.
        
        Similar a _has_speech_rms, mas retorna também o nível RMS máximo em dB
        para permitir logging e calibração do threshold.
        
        Args:
            audio_array: Array numpy de áudio normalizado (-1 a 1)
            sample_rate: Taxa de amostragem em Hz
        
        Returns:
            Tuple[bool, float]: (has_speech, max_rms_db)
                - has_speech: True se detectar fala, False caso contrário
                - max_rms_db: Nível RMS máximo encontrado em dB
        """
        if len(audio_array) == 0:
            return False, -float('inf')
        
        # Validar sample_rate antes de usar
        if sample_rate <= 0:
            logger.warn("⚠️ [PRÉ-PROCESSAMENTO] Sample rate inválido para detecção RMS", sample_rate=sample_rate)
            return False, -float('inf')
        
        # Usar threshold configurado
        rms_threshold_db = self._rms_speech_threshold_db
        window_size_ms = 100
        
        # Converter threshold dB para linear (RMS)
        rms_threshold_linear = 10 ** (rms_threshold_db / 20.0)
        
        # Calcular tamanho da janela em samples
        window_samples = int(sample_rate * window_size_ms / 1000.0)
        if window_samples < 1:
            window_samples = 1
        
        max_rms = self._rms_max_over_windows(audio_array, window_samples)
        # Converter para dB
        max_rms_db = 20 * np.log10(max_rms + 1e-10)
        has_speech = max_rms >= rms_threshold_linear
        
        return has_speech, max_rms_db
    
    def _trim_silence(self, audio_array: np.ndarray, sample_rate: int,
                     silence_threshold_db: float = -35.0, frame_length_ms: int = 50,
                     hop_length_ms: int = 25):
        """
        P1.3: Remove silêncio inicial e final do áudio.
        
        Remove períodos de silêncio que causam alucinações do Whisper.
        Mantém padding mínimo para preservar contexto de fala.
        
        Args:
            audio_array: Array numpy de áudio normalizado (-1 a 1)
            sample_rate: Taxa de amostragem em Hz
            silence_threshold_db: Threshold em dB para considerar silêncio (padrão -35dB)
            frame_length_ms: Tamanho do frame em ms
            hop_length_ms: Tamanho do hop em ms
        
        Returns:
            Tuple (audio_trimmed, info_dict) onde info_dict contém estatísticas do trim
        """
        if len(audio_array) == 0:
            return audio_array, {'trimmed_start_sec': 0.0, 'trimmed_end_sec': 0.0, 'original_length_sec': 0.0}
        
        # Validar sample_rate antes de usar
        if sample_rate <= 0:
            logger.warn("⚠️ [PRÉ-PROCESSAMENTO] Sample rate inválido para trim", sample_rate=sample_rate)
            return audio_array, {'trimmed_start_sec': 0.0, 'trimmed_end_sec': 0.0, 'original_length_sec': 0.0}
        
        original_length = len(audio_array)
        original_duration_sec = original_length / sample_rate
        
        # Converter threshold dB para linear
        silence_threshold_linear = 10 ** (silence_threshold_db / 20.0)
        
        # Calcular tamanhos em samples
        frame_samples = int(sample_rate * frame_length_ms / 1000.0)
        hop_samples = int(sample_rate * hop_length_ms / 1000.0)
        
        if frame_samples < 1:
            frame_samples = 1
        if hop_samples < 1:
            hop_samples = 1
        
        # Encontrar início da fala (primeira janela com RMS acima do threshold)
        start_idx = 0
        for i in range(0, len(audio_array) - frame_samples, hop_samples):
            window = audio_array[i:i + frame_samples]
            rms = np.sqrt(np.mean(window ** 2))
            if rms >= silence_threshold_linear:
                # Encontrar início: voltar até encontrar silêncio ou início
                # Procurar retroativamente por padding mínimo (100ms)
                padding_samples = int(sample_rate * 0.1)  # 100ms padding
                start_idx = max(0, i - padding_samples)
                break
        
        # Encontrar fim da fala (última janela com RMS acima do threshold)
        end_idx = len(audio_array)
        for i in range(len(audio_array) - frame_samples, 0, -hop_samples):
            window = audio_array[i:i + frame_samples]
            rms = np.sqrt(np.mean(window ** 2))
            if rms >= silence_threshold_linear:
                # Encontrar fim: avançar até encontrar silêncio ou fim
                # Procurar prospectivamente por padding mínimo (100ms)
                padding_samples = int(sample_rate * 0.1)  # 100ms padding
                end_idx = min(len(audio_array), i + frame_samples + padding_samples)
                break
        
        # Se não encontrou fala, manter mínimo (não remover tudo)
        if end_idx <= start_idx:
            # Manter pelo menos 100ms do centro
            center = len(audio_array) // 2
            min_samples = int(sample_rate * 0.1)
            start_idx = max(0, center - min_samples // 2)
            end_idx = min(len(audio_array), center + min_samples // 2)
        
        # Aplicar trim
        audio_trimmed = audio_array[start_idx:end_idx]
        
        trimmed_start_sec = start_idx / sample_rate
        trimmed_end_sec = (len(audio_array) - end_idx) / sample_rate
        trimmed_length_sec = len(audio_trimmed) / sample_rate
        
        trim_info = {
            'trimmed_start_sec': round(trimmed_start_sec, 3),
            'trimmed_end_sec': round(trimmed_end_sec, 3),
            'original_length_sec': round(original_duration_sec, 3),
            'trimmed_length_sec': round(trimmed_length_sec, 3),
            'samples_removed_start': start_idx,
            'samples_removed_end': len(audio_array) - end_idx
        }
        
        logger.debug(
            "✂️ [PRÉ-PROCESSAMENTO] Trim de silêncio aplicado",
            **trim_info
        )
        
        return audio_trimmed, trim_info
    
    def _estimate_snr(self, audio_array: np.ndarray, sample_rate: int,
                     frame_length_ms: int = 100) -> float:
        """
        P2.2: Estima SNR (Signal-to-Noise Ratio) do áudio usando análise de energia.
        
        Calcula diferença entre energia em regiões de fala vs silêncio.
        Usado para decidir se VAD pode ser usado com segurança.
        
        Args:
            audio_array: Array numpy de áudio normalizado (-1 a 1)
            sample_rate: Taxa de amostragem em Hz
            frame_length_ms: Tamanho do frame em ms para análise
        
        Returns:
            SNR estimado em dB (pode ser negativo se muito ruidoso)
        """
        if len(audio_array) == 0:
            return -999.0  # SNR muito baixo (inválido)
        
        # Validar sample_rate antes de usar
        if sample_rate <= 0:
            return -999.0  # SNR inválido
        
        frame_samples = int(sample_rate * frame_length_ms / 1000.0)
        if frame_samples < 1:
            frame_samples = 1
        
        # Calcular energia RMS por frame
        frame_energies = []
        for i in range(0, len(audio_array), frame_samples):
            frame = audio_array[i:i + frame_samples]
            if len(frame) == 0:
                break
            rms = np.sqrt(np.mean(frame ** 2))
            frame_energies.append(rms)
        
        if len(frame_energies) < 2:
            return -999.0
        
        frame_energies = np.array(frame_energies)
        
        # Estimar energia de sinal (frames com maior energia - percentil 75)
        # Estimar energia de ruído (frames com menor energia - percentil 25)
        signal_energy = np.percentile(frame_energies, 75)
        noise_energy = np.percentile(frame_energies, 25)
        
        # Evitar divisão por zero
        if noise_energy < 1e-10:
            # Se ruído muito baixo, assumir SNR alto
            if signal_energy > 1e-10:
                return 30.0  # SNR alto (limite superior)
            else:
                return -999.0  # Áudio muito quieto
        
        # SNR = 20 * log10(signal_rms / noise_rms)
        snr_db = 20 * np.log10(signal_energy / noise_energy)
        
        # Limitar valores extremos
        snr_db = max(-40.0, min(40.0, snr_db))
        
        return float(snr_db)
    
    def _preprocess_audio_sync(
        self,
        audio_data: bytes,
        sample_rate: int,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fase 1: Pré-processamento de áudio executado em thread separada (síncrono).
        
        NOVO: Funciona como CLASSIFICADOR, não como FILTRO BLOQUEANTE.
        Sempre retorna resultado, com has_speech flag indicando se áudio tem fala.
        
        Encapsula todas as operações bloqueantes de pré-processamento:
        - Decodificação WAV
        - Validações de áudio
        - Detecção de fala (RMS) - agora como classificador
        - Trim de silêncio (apenas se has_speech=True)
        - Estimação de SNR (apenas se has_speech=True)
        - Cálculo de parâmetros para transcrição (apenas se has_speech=True)    
        
        Esta função é executada no ThreadPoolExecutor para não bloquear o event loop.
        
        Args:
            audio_data: Bytes do arquivo WAV (incluindo header)
            sample_rate: Taxa de amostragem do áudio (Hz)
            language: Idioma do áudio (opcional)
        
        Returns:
            Dict SEMPRE retornado (nunca None):
            
            Se has_speech=False (áudio sem fala):
            {
                'has_speech': False,
                'rejection_reason': str,  # 'decode_failed', 'too_short', 'no_speech_detected'
                'rms_max_db': float,
                'audio_duration_sec': float,
                'rms_threshold_db': float
            }
            
            Se has_speech=True (áudio com fala):
            {
                'has_speech': True,
                'audio_array': np.ndarray,
                'sample_rate': int,
                'trim_info': dict,
                'estimated_snr_db': float,
                'beam_size': int,
                'use_vad': bool,
                'audio_duration_sec_after_trim': float,
                'transcribe_options': dict,
                'rms_max_db': float
            }
        """
        try:
            # Decodificar WAV para array numpy
            # O Whisper espera áudio como array numpy float32 normalizado (-1 a 1)
            audio_array = self._decode_wav(audio_data, sample_rate)
            
            if audio_array is None or len(audio_array) == 0:
                logger.warn(
                    "⚠️ [PRÉ-PROCESSAMENTO] Falha na decodificação WAV ou áudio vazio",
                    audio_size_bytes=len(audio_data),
                    sample_rate=sample_rate,
                    reason="decode_wav retornou None ou array vazio"
                )
                return {
                    'has_speech': False,
                    'rejection_reason': 'decode_failed',
                    'rms_max_db': -float('inf'),
                    'audio_duration_sec': 0.0,
                    'rms_threshold_db': self._rms_speech_threshold_db
                }
            
            # Filtrar chunks muito pequenos
            audio_duration_sec = len(audio_array) / sample_rate
            if audio_duration_sec < self._min_audio_duration_sec:
                logger.debug(
                    "⏭️ [PRÉ-PROCESSAMENTO] Áudio muito curto (antes de trim)",
                    audio_duration_sec=round(audio_duration_sec, 3),
                    min_duration_sec=self._min_audio_duration_sec,
                    reason="Áudio menor que mínimo antes de trim"
                )
                return {
                    'has_speech': False,
                    'rejection_reason': 'too_short',
                    'rms_max_db': -float('inf'),
                    'audio_duration_sec': audio_duration_sec,
                    'rms_threshold_db': self._rms_speech_threshold_db
                }
            
            # P1.2: Detecção de fala RMS - AGORA COMO CLASSIFICADOR, NÃO FILTRO BLOQUEANTE
            # Classifica áudio rapidamente como "fala" ou "silêncio" sem bloquear pipeline
            has_speech, max_rms_db = self._has_speech_rms_with_level(audio_array, sample_rate)
            
            if not has_speech:
                # Áudio classificado como silêncio - retornar imediatamente SEM processamento pesado
                # Pipeline continua fluindo, próximo áudio será processado sem espera
                logger.debug(
                    "⏭️ [CLASSIFICADOR] RMS classificou como silêncio",
                    audio_duration_sec=round(audio_duration_sec, 3),
                    rms_max_db=round(max_rms_db, 2),
                    rms_threshold_db=self._rms_speech_threshold_db,
                    reason="RMS abaixo do threshold - áudio descartado rapidamente"
                )
                return {
                    'has_speech': False,
                    'rejection_reason': 'no_speech_detected',
                    'rms_max_db': max_rms_db,
                    'audio_duration_sec': audio_duration_sec,
                    'rms_threshold_db': self._rms_speech_threshold_db
                }
            
            # Se chegou aqui: has_speech=True, continuar com pré-processamento completo
            
            # P1.3: Trim de silêncio inicial/final - remove silêncio que causa alucinações
            audio_array, trim_info = self._trim_silence(audio_array, sample_rate)
            
            # REVISADO P4.2: Verificar se após trim ainda há áudio suficiente para transcrição confiável
            # Reduzido de 1.5s para 0.8s para permitir transcrições de áudio mais curto
            # 1.5s era muito restritivo e rejeitava áudio válido (ex: respostas curtas)
            audio_duration_sec_after_trim = len(audio_array) / sample_rate
            min_audio_after_trim_sec = 0.8  # REVISADO: Reduzido de 1.5s para 0.8s (era 0.5s antes de P4.2)
            if audio_duration_sec_after_trim < min_audio_after_trim_sec:
                logger.debug(
                    "⏭️ [CLASSIFICADOR] Áudio muito curto após trim",
                    audio_duration_sec_before_trim=round(audio_duration_sec, 3),
                    audio_duration_sec_after_trim=round(audio_duration_sec_after_trim, 3),
                    min_audio_after_trim_sec=min_audio_after_trim_sec,
                    trim_start_sec=trim_info.get('trimmed_start_sec', 0),
                    trim_end_sec=trim_info.get('trimmed_end_sec', 0),
                    reason=f"Áudio após trim ({audio_duration_sec_after_trim:.2f}s) menor que mínimo ({min_audio_after_trim_sec}s)"
                )
                return {
                    'has_speech': False,
                    'rejection_reason': 'too_short_after_trim',
                    'rms_max_db': max_rms_db,
                    'audio_duration_sec': audio_duration_sec,
                    'audio_duration_sec_after_trim': audio_duration_sec_after_trim,
                    'rms_threshold_db': self._rms_speech_threshold_db
                }
            
            # P2.2: Estimar SNR para VAD seletivo e ajuste de parâmetros
            estimated_snr_db = self._estimate_snr(audio_array, sample_rate)
            
            # P2.3: Ajustar beam_size conforme duração do áudio
            # Áudio mais longo se beneficia de beam search maior
            beam_size = 7 if audio_duration_sec_after_trim >= 10.0 else 5
            
            # P2.2: VAD seletivo - ativar apenas se SNR for suficientemente alto
            # SNR alto = áudio limpo = VAD seguro. SNR baixo = risco de remover fala válida
            use_vad = estimated_snr_db > 10.0  # Threshold de 10dB para considerar áudio "limpo"
            
            # P2.1: Ajustar thresholds baseado em qualidade de áudio para reduzir alucinações
            # no_speech_threshold: 0.5 (mais restritivo que 0.3, menos alucinações)
            # log_prob_threshold: -1.0 (menos restritivo que -0.8, não corta fala válida)
            # compression_ratio_threshold: 2.4 (menos agressivo que 2.0, permite repetições naturais)
            transcribe_options = {
                'language': language or self.language,
                'task': self.task,  # 'transcribe' ou 'translate'
                'temperature': 0.0,  # Temperatura 0 = mais determinístico e preciso
                'condition_on_previous_text': False,  # Evitar repetições quando texto anterior é ruim
                'compression_ratio_threshold': 2.4,  # P2.1: Menos agressivo (era 2.0) - permite repetições naturais
                'log_prob_threshold': -1.0,  # P2.1: Menos restritivo (era -0.8) - não corta fala válida
                'no_speech_threshold': 0.5,  # P2.1: Mais restritivo (era 0.3) - reduz alucinações em silêncio
                'beam_size': beam_size,  # P2.3: Dinâmico conforme duração (5 para <10s, 7 para ≥10s)
                'vad_filter': use_vad,  # P2.2: Seletivo baseado em SNR estimado
            }
            
            # Retornar resultado com has_speech=True e todos os dados de pré-processamento
            return {
                'has_speech': True,
                'audio_array': audio_array,
                'sample_rate': sample_rate,
                'trim_info': trim_info,
                'estimated_snr_db': estimated_snr_db,
                'beam_size': beam_size,
                'use_vad': use_vad,
                'audio_duration_sec_after_trim': audio_duration_sec_after_trim,
                'transcribe_options': transcribe_options,
                'rms_max_db': max_rms_db
            }
        except Exception as e:
            logger.error(
                "❌ [PRÉ-PROCESSAMENTO] Exceção durante pré-processamento",
                error=str(e),
                error_type=type(e).__name__,
                audio_size_bytes=len(audio_data)
            )
            # Retornar resultado indicando falha
            return {
                'has_speech': False,
                'rejection_reason': 'exception',
                'rms_max_db': -float('inf'),
                'audio_duration_sec': 0.0,
                'rms_threshold_db': self._rms_speech_threshold_db,
                'error': str(e)
            }
    
    def _fix_common_transcription_errors(self, text: str) -> str:
        """
        FASE 3: Corrige erros comuns de transcrição do Whisper em português.
        
        Corrige erros de transcrição frequentes onde o Whisper separa incorretamente
        palavras compostas (ex: "a diário" → "adiar").
        
        Esta função apenas corrige erros, não rejeita texto, garantindo que
        feedbacks válidos não sejam bloqueados.
        
        Args:
            text: Texto transcrito a ser corrigido
        
        Returns:
            Texto corrigido
        """
        if not text:
            return text
        
        original_text = text
        
        # Correções de palavras comuns (erros frequentes do Whisper em português)
        corrections = [
            # Erro: "preciso a diário" → correto: "preciso adiar"
            (r'\bpreciso\s+a\s+diário\b', 'preciso adiar', re.IGNORECASE),
            # Erro: "a diário" → correto: "adiar"
            (r'\ba\s+diário\b', 'adiar', re.IGNORECASE),
            # Erro: "a mar" → correto: "amar"
            (r'\ba\s+mar\b', 'amar', re.IGNORECASE),
            # Adicionar mais correções conforme necessário
        ]
        
        corrected_text = text
        for pattern, replacement, flags in corrections:
            corrected_text = re.sub(pattern, replacement, corrected_text, flags=flags)
        
        if corrected_text != original_text:
            logger.info(
                "🔧 [FASE 3] Texto corrigido (erros comuns de transcrição)",
                original=original_text[:100],
                corrected=corrected_text[:100],
                note="Correção automática de erros comuns sem bloquear feedbacks"
            )
        
        return corrected_text

