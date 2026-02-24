"""
Pipeline de pré-processamento de áudio para transcrição.
Orquestra decode WAV, detecção de fala (RMS), trim de silêncio e estimação de SNR.
Recebe config e callables injetados (ex.: pelo TranscriptionService).
"""

import numpy as np
import structlog
from typing import Optional, Dict, Any, Callable, Tuple

logger = structlog.get_logger()


class PreprocessPipeline:
    """
    Pipeline de pré-processamento: decode, validação, classificação RMS, trim, SNR, opções.
    Executado de forma síncrona (tipicamente em ThreadPoolExecutor).
    """

    def __init__(
        self,
        *,
        min_audio_duration_sec: float,
        rms_speech_threshold_db: float,
        language: str,
        task: str,
        min_audio_after_trim_sec: float = 0.8,
        decode_wav: Callable[[bytes, int], Optional[np.ndarray]],
        has_speech_rms_with_level: Callable[[np.ndarray, int], Tuple[bool, float]],
        trim_silence: Callable[[np.ndarray, int], Tuple[np.ndarray, Dict[str, Any]]],
        estimate_snr: Callable[[np.ndarray, int], float],
    ) -> None:
        self.min_audio_duration_sec = min_audio_duration_sec
        self.rms_speech_threshold_db = rms_speech_threshold_db
        self.language = language
        self.task = task
        self.min_audio_after_trim_sec = min_audio_after_trim_sec
        self.decode_wav = decode_wav
        self.has_speech_rms_with_level = has_speech_rms_with_level
        self.trim_silence = trim_silence
        self.estimate_snr = estimate_snr

    def run(
        self,
        audio_data: bytes,
        sample_rate: int,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
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
            audio_array = self.decode_wav(audio_data, sample_rate)

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
                    'rms_threshold_db': self.rms_speech_threshold_db
                }

            # Filtrar chunks muito pequenos
            audio_duration_sec = len(audio_array) / sample_rate
            if audio_duration_sec < self.min_audio_duration_sec:
                logger.debug(
                    "⏭️ [PRÉ-PROCESSAMENTO] Áudio muito curto (antes de trim)",
                    audio_duration_sec=round(audio_duration_sec, 3),
                    min_duration_sec=self.min_audio_duration_sec,
                    reason="Áudio menor que mínimo antes de trim"
                )
                return {
                    'has_speech': False,
                    'rejection_reason': 'too_short',
                    'rms_max_db': -float('inf'),
                    'audio_duration_sec': audio_duration_sec,
                    'rms_threshold_db': self.rms_speech_threshold_db
                }

            # Classifica áudio rapidamente como "fala" ou "silêncio" sem bloquear pipeline
            has_speech, max_rms_db = self.has_speech_rms_with_level(audio_array, sample_rate)

            if not has_speech:
                # Áudio classificado como silêncio - retornar imediatamente SEM processamento pesado
                # Pipeline continua fluindo, próximo áudio será processado sem espera
                logger.debug(
                    "⏭️ [CLASSIFICADOR] RMS classificou como silêncio",
                    audio_duration_sec=round(audio_duration_sec, 3),
                    rms_max_db=round(max_rms_db, 2),
                    rms_threshold_db=self.rms_speech_threshold_db,
                    reason="RMS abaixo do threshold - áudio descartado rapidamente"
                )
                return {
                    'has_speech': False,
                    'rejection_reason': 'no_speech_detected',
                    'rms_max_db': max_rms_db,
                    'audio_duration_sec': audio_duration_sec,
                    'rms_threshold_db': self.rms_speech_threshold_db
                }

            # Se chegou aqui: has_speech=True, continuar com pré-processamento completo
            audio_array, trim_info = self.trim_silence(audio_array, sample_rate)

            audio_duration_sec_after_trim = len(audio_array) / sample_rate
            if audio_duration_sec_after_trim < self.min_audio_after_trim_sec:
                logger.debug(
                    "⏭️ [CLASSIFICADOR] Áudio muito curto após trim",
                    audio_duration_sec_before_trim=round(audio_duration_sec, 3),
                    audio_duration_sec_after_trim=round(audio_duration_sec_after_trim, 3),
                    min_audio_after_trim_sec=self.min_audio_after_trim_sec,
                    trim_start_sec=trim_info.get('trimmed_start_sec', 0),
                    trim_end_sec=trim_info.get('trimmed_end_sec', 0),
                    reason=f"Áudio após trim ({audio_duration_sec_after_trim:.2f}s) menor que mínimo ({self.min_audio_after_trim_sec}s)"
                )
                return {
                    'has_speech': False,
                    'rejection_reason': 'too_short_after_trim',
                    'rms_max_db': max_rms_db,
                    'audio_duration_sec': audio_duration_sec,
                    'audio_duration_sec_after_trim': audio_duration_sec_after_trim,
                    'rms_threshold_db': self.rms_speech_threshold_db
                }

            estimated_snr_db = self.estimate_snr(audio_array, sample_rate)

            # Áudio mais longo se beneficia de beam search maior
            beam_size = 7 if audio_duration_sec_after_trim >= 10.0 else 5

            # SNR alto = áudio limpo = VAD seguro. SNR baixo = risco de remover fala válida
            use_vad = estimated_snr_db > 10.0  # Threshold de 10dB para considerar áudio "limpo"

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
                'rms_threshold_db': self.rms_speech_threshold_db,
                'error': str(e)
            }
