import asyncio
import time
import zlib
from typing import Optional

import structlog

from .circular_buffer import CircularAudioBuffer
from .codec import pcm_int16_to_wav
from .types import AudioKey, BufferReadyCallback

logger = structlog.get_logger()


class SlidingWindowWorker:
    """Worker para uma chave de stream de áudio.

    Lê janelas de áudio do CircularAudioBuffer (que é alimentado por quem envia
    os chunks) e envia cada janela para transcrição através do callback injetado.
    O callback (ex.: on_buffer_ready no socketio_server) recebe o WAV e dispara
    transcrição e análise; esta classe apenas entrega o áudio para esse callback.
    """

    def __init__(
        self,
        key: AudioKey,
        ring: CircularAudioBuffer,
        callback: BufferReadyCallback,
        window_sec: float,
        hop_sec: float,
        tick_sec: float,
        min_sleep_sec: float = 0.05,
    ):
        self.key = key
        self.ring = ring
        self.callback = callback
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.tick_sec = tick_sec
        self.min_sleep_sec = min_sleep_sec
        self.window_samples = max(1, int(window_sec * ring.sample_rate))
        self.hop_samples = max(1, int(hop_sec * ring.sample_rate))
        self.last_processed_end_sample = 0
        self.last_window_crc: Optional[int] = None
        self.skipped_cycles = 0
        self._stop = asyncio.Event()
        self._stopped = asyncio.Event()

    def request_stop(self):
        self._stop.set()

    async def wait_stopped(self, timeout_sec: float = 2.0):
        try:
            await asyncio.wait_for(self._stopped.wait(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            logger.warn(
                "⚠️ [CONTINUOUS_WORKER] Worker did not stop in time",
                key=self.key,
                timeout_sec=timeout_sec,
            )

    async def run(self):
        while not self._stop.is_set():
            cycle_start = time.perf_counter()
            try:
                await self._run_cycle()
            except Exception as exc:
                logger.error(
                    "❌ [CONTINUOUS_WORKER] Worker cycle failed",
                    key=self.key,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
            elapsed = time.perf_counter() - cycle_start
            await asyncio.sleep(max(self.min_sleep_sec, self.tick_sec - elapsed))
        self._stopped.set()

    async def _run_cycle(self):
        if not self.ring.has_enough_audio(self.window_sec):
            self.skipped_cycles += 1
            return

        written_total = self.ring.get_written_samples_total()
        if written_total - self.last_processed_end_sample < self.hop_samples:
            self.skipped_cycles += 1
            return

        extract_start = time.perf_counter()
        pcm, end_sample = self.ring.snapshot_last_samples(self.window_samples)
        extract_ms = (time.perf_counter() - extract_start) * 1000
        if pcm is None or pcm.size == 0:
            self.skipped_cycles += 1
            return
        if end_sample <= self.last_processed_end_sample:
            self.skipped_cycles += 1
            return

        window_crc = zlib.crc32(pcm.tobytes())
        if self.last_window_crc is not None and window_crc == self.last_window_crc:
            self.last_processed_end_sample = end_sample
            self.skipped_cycles += 1
            return

        callback_start = time.perf_counter()
        wav_data = pcm_int16_to_wav(pcm, self.ring.sample_rate, channels=1)
        await self.callback(
            meeting_id=self.key[0],
            participant_id=self.key[1],
            track=self.key[2],
            wav_data=wav_data,
            sample_rate=self.ring.sample_rate,
            channels=1,
            timestamp=int(time.time() * 1000),
        )
        callback_ms = (time.perf_counter() - callback_start) * 1000

        self.last_window_crc = window_crc
        self.last_processed_end_sample = end_sample

        logger.debug(
            "[CONTINUOUS_WORKER] Cycle metrics",
            key=self.key,
            extract_ms=round(extract_ms, 2),
            callback_ms=round(callback_ms, 2),
            cycle_total_ms=round(extract_ms + callback_ms, 2),
            window_crc=window_crc,
            last_processed_end_sample=self.last_processed_end_sample,
            skipped_cycles=self.skipped_cycles,
        )

