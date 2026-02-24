import threading
from typing import Optional, Tuple

import numpy as np

from .codec import wav_to_pcm_int16


class CircularAudioBuffer:
    """Buffer circular de áudio.

    Basicamente coloca o áudio no buffer: append_wav/append_pcm gravam no ring.
    Quando o buffer enche, os dados mais antigos são sobrescritos (ring buffer).
    Também permite consultar e ler sem consumir: has_enough_audio, get_written_samples_total
    e snapshot_last_samples (cópia das últimas N amostras para transcrição, etc.).
    """

    def __init__(self, sample_rate: int, channels: int, capacity_sec: float):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if channels <= 0:
            raise ValueError("channels must be > 0")
        self.sample_rate = sample_rate
        self.channels = channels
        self.capacity_samples = max(1, int(capacity_sec * sample_rate))
        self._ring = np.zeros(self.capacity_samples, dtype=np.int16)
        self._write_pos = 0
        self._written_samples_total = 0
        self._lock = threading.Lock()

    def append_wav(self, wav_data: bytes) -> int:
        pcm = wav_to_pcm_int16(wav_data)
        return self.append_pcm(pcm)

    def append_pcm(self, pcm: np.ndarray) -> int:
        if pcm.size == 0:
            return 0
        pcm = pcm.astype(np.int16, copy=False)
        with self._lock:
            remaining = pcm.size
            src_idx = 0
            while remaining > 0:
                space_until_wrap = self.capacity_samples - self._write_pos
                chunk = min(remaining, space_until_wrap)
                self._ring[self._write_pos : self._write_pos + chunk] = pcm[src_idx : src_idx + chunk]
                self._write_pos = (self._write_pos + chunk) % self.capacity_samples
                src_idx += chunk
                remaining -= chunk
            self._written_samples_total += pcm.size
        return pcm.size

    def get_written_samples_total(self) -> int:
        with self._lock:
            return self._written_samples_total

    def has_enough_audio(self, seconds: float) -> bool:
        needed = max(1, int(seconds * self.sample_rate))
        with self._lock:
            available = min(self._written_samples_total, self.capacity_samples)
            return available >= needed

    def snapshot_last_samples(self, sample_count: int) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            available = min(self._written_samples_total, self.capacity_samples)
            if available <= 0:
                return None, self._written_samples_total
            sample_count = max(1, min(sample_count, available))
            end_pos = self._write_pos
            start_pos = (end_pos - sample_count) % self.capacity_samples
            if start_pos < end_pos:
                data = self._ring[start_pos:end_pos].copy()
            else:
                data = np.concatenate((self._ring[start_pos:], self._ring[:end_pos])).copy()
            end_sample = self._written_samples_total
        return data, end_sample

