from typing import Optional

import numpy as np
import structlog

from .codec import pcm_int16_to_wav, wav_to_pcm_int16

logger = structlog.get_logger()


class AudioBuffer:
    """Legacy non-continuous buffer (fallback mode)."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_chunks: list[bytes] = []
        self.first_timestamp: Optional[int] = None
        self.last_timestamp: Optional[int] = None
        self.total_samples = 0

    def add_chunk(self, wav_data: bytes, timestamp: int):
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        self.last_timestamp = timestamp
        self.audio_chunks.append(wav_data)
        if len(wav_data) > 44:
            self.total_samples += (len(wav_data) - 44) // 2

    def get_duration_sec(self) -> float:
        if self.total_samples == 0:
            return 0.0
        return self.total_samples / self.sample_rate

    def get_combined_wav(self) -> bytes:
        if not self.audio_chunks:
            return b""
        all_pcm_data = []
        for wav_chunk in self.audio_chunks:
            try:
                pcm = wav_to_pcm_int16(wav_chunk)
                if pcm.size > 0:
                    all_pcm_data.append(pcm)
            except Exception as e:
                logger.warn(
                    "⚠️ [BUFFER] Erro ao decodificar chunk WAV",
                    error=str(e),
                    chunk_size=len(wav_chunk),
                )
                continue
        if not all_pcm_data:
            return b""
        combined_pcm = np.concatenate(all_pcm_data)
        return pcm_int16_to_wav(combined_pcm, self.sample_rate, channels=self.channels)

    def clear(self):
        self.audio_chunks.clear()
        self.first_timestamp = None
        self.last_timestamp = None
        self.total_samples = 0

