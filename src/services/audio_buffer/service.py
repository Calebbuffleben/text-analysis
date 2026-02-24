import asyncio
import os
import time
from typing import Dict, Optional

import structlog

from ..facades.audio_buffer_facade import AudioBufferFacade
from .legacy_buffer import AudioBuffer
from .stream_manager import ContinuousAudioStreamManager
from .types import AudioKey, BufferReadyCallback

logger = structlog.get_logger()


class AudioBufferService(AudioBufferFacade):
    """Facade que escolhe entre o modo contínuo circular e o modo legado."""

    def __init__(
        self,
        min_duration_sec: float = 3.0,
        max_duration_sec: float = 10.0,
        flush_interval_sec: float = 2.0,
    ):
        """Guarda min/max de duração e intervalo de flush (legado) e lê config do modo contínuo.
        Se contínuo ativo, cria o stream manager; senão usa só buffers legados."""
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.flush_interval_sec = flush_interval_sec
        self.buffers: Dict[AudioKey, AudioBuffer] = {}
        self.flush_timers: Dict[AudioKey, asyncio.Task] = {}
        self.on_buffer_ready: Optional[BufferReadyCallback] = None

        self.continuous_enabled = os.getenv("CONTINUOUS_WORKER_ENABLED", "false").lower() == "true"
        self.ring_capacity_sec = float(os.getenv("CONTINUOUS_RING_CAPACITY_SEC", "20"))
        self.window_sec = float(os.getenv("CONTINUOUS_WINDOW_SEC", "5"))
        self.hop_sec = float(os.getenv("CONTINUOUS_HOP_SEC", "1"))
        self.tick_sec = float(os.getenv("CONTINUOUS_TICK_SEC", "1"))
        self.stream_manager: Optional[ContinuousAudioStreamManager] = None
        if self.continuous_enabled:
            self.stream_manager = ContinuousAudioStreamManager(
                callback=None,
                ring_capacity_sec=self.ring_capacity_sec,
                window_sec=self.window_sec,
                hop_sec=self.hop_sec,
                tick_sec=self.tick_sec,
            )

        logger.info(
            "✅ [BUFFER] AudioBufferService initialized",
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
            flush_interval_sec=flush_interval_sec,
            continuous_enabled=self.continuous_enabled,
            ring_capacity_sec=self.ring_capacity_sec,
            window_sec=self.window_sec,
            hop_sec=self.hop_sec,
            tick_sec=self.tick_sec,
        )

    def set_callback(self, callback: BufferReadyCallback):
        """Define quem recebe o áudio pronto (ex.: transcrição). No modo contínuo repassa ao stream manager."""
        self.on_buffer_ready = callback
        if self.stream_manager is not None:
            self.stream_manager.set_callback(callback)

    async def add_chunk(
        self,
        meeting_id: str,
        participant_id: str,
        track: str,
        wav_data: bytes,
        sample_rate: int,
        channels: int,
        timestamp: int,
    ) -> Optional[bytes]:
        """Recebe um chunk. Contínuo: repassa ao stream manager.
        Legado: acumula no buffer; se passou min/max de duração faz flush na hora; senão agenda flush após o intervalo."""
        if self.continuous_enabled and self.stream_manager is not None:
            await self.stream_manager.push_chunk(
                meeting_id=meeting_id,
                participant_id=participant_id,
                track=track,
                wav_data=wav_data,
                sample_rate=sample_rate,
                channels=channels,
            )
            return None

        key = (meeting_id, participant_id, track)
        if key not in self.buffers:
            self.buffers[key] = AudioBuffer(sample_rate=sample_rate, channels=channels)
        buffer = self.buffers[key]
        buffer.add_chunk(wav_data, timestamp)
        if key in self.flush_timers:
            self.flush_timers[key].cancel()
        duration = buffer.get_duration_sec()
        # Legacy behavior: flush as soon as minimum duration is reached.
        # max_duration remains as configuration compatibility in this mode.
        if duration >= self.min_duration_sec:
            return self._flush_buffer(key)
        self.flush_timers[key] = asyncio.create_task(self._schedule_flush(key, self.flush_interval_sec))
        return None

    async def _schedule_flush(self, key: AudioKey, delay_sec: float):
        """LEGADO. Após o delay, junta todo o áudio do buffer daquele stream num WAV e chama o callback com ele."""
        try:
            await asyncio.sleep(delay_sec)
            if key in self.buffers and self.on_buffer_ready:
                buffer = self.buffers[key]
                if buffer.get_duration_sec() > 0:
                    combined_wav = self._flush_buffer(key)
                    if combined_wav:
                        await self.on_buffer_ready(
                            meeting_id=key[0],
                            participant_id=key[1],
                            track=key[2],
                            wav_data=combined_wav,
                            sample_rate=buffer.sample_rate,
                            channels=buffer.channels,
                            timestamp=buffer.last_timestamp or int(time.time() * 1000),
                        )
        except asyncio.CancelledError:
            return

    def _flush_buffer(self, key: AudioKey) -> Optional[bytes]:
        """LEGADO. Junta os chunks do buffer num WAV, limpa o buffer, remove o timer e devolve esse WAV."""
        if key not in self.buffers:
            return None
        buffer = self.buffers[key]
        if not buffer.audio_chunks:
            return None
        combined_wav = buffer.get_combined_wav()
        buffer.clear()
        del self.buffers[key]
        if key in self.flush_timers:
            self.flush_timers[key].cancel()
            del self.flush_timers[key]
        return combined_wav

    async def stop_stream(self, meeting_id: str, participant_id: str, track: str):
        """Para um stream: no contínuo para o worker e remove buffer; no legado remove buffer e cancela o timer."""
        key = (meeting_id, participant_id, track)
        if self.stream_manager is not None:
            await self.stream_manager.stop_stream(key)
        if key in self.buffers:
            self.buffers[key].clear()
            del self.buffers[key]
        if key in self.flush_timers:
            self.flush_timers[key].cancel()
            del self.flush_timers[key]

    async def shutdown(self):
        """Desliga tudo: para todos os workers (contínuo), cancela todos os timers (legado) e esvazia os buffers."""
        if self.stream_manager is not None:
            await self.stream_manager.stop_all()
        for timer in self.flush_timers.values():
            timer.cancel()
        self.flush_timers.clear()
        self.buffers.clear()

