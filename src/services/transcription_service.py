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
import structlog
from faster_whisper import WhisperModel
import numpy as np
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
        O modelo Whisper será carregado apenas na primeira transcrição (lazy loading).
        """
        self.model = None
        self._loaded = False
        self.model_name = Config.WHISPER_MODEL_NAME
        self.device = Config.WHISPER_DEVICE
        self.language = Config.WHISPER_LANGUAGE
        self.task = Config.WHISPER_TASK
        
        # faster-whisper compute_type: "int8" para CPU (mais rápido), "float16" para GPU
        import os
        compute_type_env = os.getenv('WHISPER_COMPUTE_TYPE', '')
        if compute_type_env:
            self.compute_type = compute_type_env
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
        # faster-whisper é mais eficiente, mas ainda limitamos a 1 transcrição por vez
        # para evitar sobrecarga e garantir que cada transcrição tenha recursos completos
        try:
            self._transcription_semaphore = asyncio.Semaphore(1)
        except RuntimeError:
            # Se não houver event loop, criar None e inicializar depois
            self._transcription_semaphore = None
        self._active_transcriptions = 0
        
        # Duração mínima de áudio para transcrição (em segundos)
        # Chunks muito pequenos (< 0.5s) são ignorados pois:
        # 1. faster-whisper funciona melhor com áudio mais longo
        # 2. Reduz carga desnecessária no CPU
        # 3. Melhora qualidade da transcrição
        self._min_audio_duration_sec = 0.5
        
        # ThreadPoolExecutor para executar faster-whisper em thread separada
        # faster-whisper é mais rápido mas ainda bloqueante, então executamos em thread separada
        # para não bloquear o event loop do asyncio
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        logger.info(
            "✅ [SERVIÇO] TranscriptionService inicializado",
            model=self.model_name,
            device=self.device,
            language=self.language,
            max_concurrent_transcriptions=1,
            min_audio_duration_sec=self._min_audio_duration_sec
        )
    
    def _load_model(self):
        """
        Carrega modelo faster-whisper (lazy loading).
        
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
        """
        if self._loaded:
            logger.debug("Modelo faster-whisper já carregado", model=self.model_name)
            return
        
        logger.info(
            "🔄 [TRANSCRIÇÃO] Carregando modelo faster-whisper",
            model=self.model_name,
            device=self.device,
            compute_type=self.compute_type
        )
        
        load_start = time.perf_counter()
        
        try:
            # faster-whisper aceita "cpu" ou "cuda" diretamente
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
        # Carregar modelo de forma assíncrona se necessário
        if not self._loaded:
            logger.info("🔄 [TRANSCRIÇÃO] Modelo não carregado, carregando agora...")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self._executor,
                self._load_model
            )
            logger.info("✅ [TRANSCRIÇÃO] Modelo carregado, prosseguindo com transcrição")
        
        # Inicializar semáforo se ainda não foi criado (fallback)
        if self._transcription_semaphore is None:
            self._transcription_semaphore = asyncio.Semaphore(1)
        
        try:
            logger.debug(
                "🔍 [TRANSCRIÇÃO] Decodificando WAV",
                audio_size_bytes=len(audio_data),
                expected_sample_rate=sample_rate
            )
            
            # Decodificar WAV para array numpy
            # O Whisper espera áudio como array numpy float32 normalizado (-1 a 1)
            audio_array = self._decode_wav(audio_data, sample_rate)
            
            if audio_array is None or len(audio_array) == 0:
                logger.warn(
                    "⚠️ [TRANSCRIÇÃO] Áudio vazio ou inválido",
                    audio_size_bytes=len(audio_data)
                )
                return {
                    'text': '',
                    'language': language or self.language,
                    'segments': [],
                    'confidence': 0.0
                }
            
            # Filtrar chunks muito pequenos
            audio_duration_sec = len(audio_array) / sample_rate
            if audio_duration_sec < self._min_audio_duration_sec:
                logger.debug(
                    "⏭️ [TRANSCRIÇÃO] Chunk muito pequeno, ignorando",
                    audio_duration_sec=round(audio_duration_sec, 2),
                    min_duration_sec=self._min_audio_duration_sec,
                    audio_samples=len(audio_array)
                )
                return {
                    'text': '',
                    'language': language or self.language,
                    'segments': [],
                    'confidence': 0.0
                }
            
            logger.debug(
                "✅ [TRANSCRIÇÃO] WAV decodificado",
                audio_samples=len(audio_array),
                audio_length_sec=round(len(audio_array) / sample_rate, 2)
            )
            
            # Configurar parâmetros de transcrição para faster-whisper
            # faster-whisper tem parâmetros ligeiramente diferentes
            transcribe_options = {
                'language': language or self.language,
                'task': self.task,  # 'transcribe' ou 'translate'
                'temperature': 0.0,  # Temperatura 0 = mais determinístico e preciso
                'condition_on_previous_text': False,  # Evitar repetições quando texto anterior é ruim
                'compression_ratio_threshold': 2.4,  # Detectar e filtrar repetições
                'log_prob_threshold': -1.0,  # Filtrar segmentos com baixa confiança (note: log_prob, não logprob)
                'no_speech_threshold': 0.3,  # Threshold mais baixo (mais permissivo) - padrão era 0.6
                'beam_size': 5,  # Beam search size (padrão é 5)
                # VAD desabilitado - estava removendo todo o áudio válido
                # O VAD do faster-whisper pode ser muito agressivo com áudio de chamadas
                'vad_filter': False,  # Desabilitar VAD para evitar remoção de áudio válido
            }
            
            audio_length_sec = len(audio_array) / sample_rate
            logger.info(
                "🎙️ [TRANSCRIÇÃO] Iniciando transcrição com Whisper",
                audio_length_sec=round(audio_length_sec, 2),
                audio_samples=len(audio_array),
                sample_rate=sample_rate,
                language=transcribe_options['language'],
                model=self.model_name,
                device=self.device
            )
            
            # Transcrever áudio em thread separada para não bloquear event loop
            # Whisper é CPU/GPU intensivo e pode demorar alguns segundos
            # Usar get_running_loop() para Python 3.7+ (mais seguro)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # Fallback para get_event_loop() se não houver loop rodando
                loop = asyncio.get_event_loop()
            
            # Usar semáforo para limitar transcrições simultâneas
            # Isso evita sobrecarga do Whisper quando muitos chunks chegam ao mesmo tempo
            logger.debug(
                "🔄 [TRANSCRIÇÃO] Aguardando slot disponível para transcrição",
                audio_length_sec=round(len(audio_array) / sample_rate, 2),
                active_transcriptions=self._active_transcriptions
            )
            
            # Adquirir semáforo ANTES de qualquer processamento
            # Isso garante que apenas uma transcrição por vez seja processada
            async with self._transcription_semaphore:
                self._active_transcriptions += 1
                transcribe_start = time.perf_counter()
                result = None
                
                try:
                    # Verificar se modelo está carregado
                    if self.model is None:
                        logger.error("❌ [TRANSCRIÇÃO] Modelo faster-whisper não está carregado!")
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
                        audio_length_sec=round(len(audio_array) / sample_rate, 2),
                        model=self.model_name,
                        compute_type=self.compute_type,
                        timeout_sec=30.0
                    )
                    
                    # Criar função de transcrição para o executor
                    # faster-whisper retorna (segments, info) ao invés de dict
                    model_ref = self.model
                    audio_ref = audio_array.copy()
                    options_ref = transcribe_options.copy()
                    language_ref = language or self.language  # Capturar language no closure
                    
                    def transcribe_sync():
                        try:
                            # faster-whisper retorna (segments, info)
                            # segments é um iterador de objetos Segment
                            segments, info = model_ref.transcribe(audio_ref, **options_ref)
                            
                            # Converter segments para lista e processar
                            segments_list = list(segments)
                            
                            # Construir texto completo concatenando segmentos
                            text = " ".join(seg.text for seg in segments_list)
                            
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
                                'duration': getattr(info, 'duration', len(audio_ref) / sample_rate)
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
                        audio_length_sec=round(len(audio_array) / sample_rate, 2),
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
                return {
                    'text': '',
                    'language': language or self.language,
                    'segments': [],
                    'confidence': 0.0
                }
            
            # Extrair informações relevantes
            text = result.get('text', '').strip()
            detected_language = result.get('language', language or self.language)
            segments = result.get('segments', [])
            
            # Calcular confiança média dos segmentos
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
        Decodifica dados WAV para array numpy.
        
        O formato WAV esperado:
        - Header de 44 bytes
        - Dados PCM16LE (16-bit little-endian)
        - Mono ou estéreo
        
        Retorna:
        - Array numpy float32 normalizado (-1.0 a 1.0)
        - Taxa de amostragem ajustada se necessário
        """
        try:
            import wave
            
            # Criar arquivo WAV em memória
            wav_file = io.BytesIO(wav_data)
            
            # Ler WAV usando wave module
            with wave.open(wav_file, 'rb') as wf:
                sample_rate = wf.getframerate()
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                num_frames = wf.getnframes()
                
                # Ler dados de áudio
                audio_bytes = wf.readframes(num_frames)
                
                # Converter bytes para array numpy
                if sample_width == 2:  # 16-bit
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                elif sample_width == 4:  # 32-bit
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int32)
                else:
                    logger.warn(f"Unsupported sample width: {sample_width}")
                    return None
                
                # Converter para float32 e normalizar (-1.0 a 1.0)
                # Para int16: dividir por 32768.0
                # Para int32: dividir por 2147483648.0
                if sample_width == 2:
                    audio_float = audio_array.astype(np.float32) / 32768.0
                else:
                    audio_float = audio_array.astype(np.float32) / 2147483648.0
                
                # Converter estéreo para mono (média dos canais)
                if num_channels == 2:
                    audio_float = audio_float.reshape(-1, 2).mean(axis=1)
                
                # Resample se necessário (Whisper funciona melhor com 16kHz)
                # Nota: Se o áudio já estiver em 16kHz, não precisa resample
                if sample_rate != expected_sample_rate:
                    try:
                        from scipy import signal
                        num_samples = int(len(audio_float) * expected_sample_rate / sample_rate)
                        if num_samples > 0:
                            audio_float = signal.resample(audio_float, num_samples)
                            logger.debug(
                                "Audio resampled",
                                from_rate=sample_rate,
                                to_rate=expected_sample_rate,
                                original_samples=len(audio_array),
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

