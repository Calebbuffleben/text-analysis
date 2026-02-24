import asyncio
from typing import Dict, Optional

import anyio
import structlog

from .circular_buffer import CircularAudioBuffer
from .sliding_worker import SlidingWindowWorker
from .types import AudioKey, BufferReadyCallback

logger = structlog.get_logger()


class ContinuousAudioStreamManager:
    """Manages circular buffers and workers per stream key.

    Além de iniciar e finalizar streams, recebe os chunks de áudio, guarda no buffer
    do stream, garante 1 worker por stream que lê as janelas do buffer e chama o
    callback (que dispara transcrição/análise). Em resumo: roteia o áudio para
    o buffer correto e mantém um worker ativo por stream.
    """

    def __init__(
        self,
        callback: Optional[BufferReadyCallback],
        ring_capacity_sec: float,
        window_sec: float,
        hop_sec: float,
        tick_sec: float,
    ):
        self.callback = callback
        self.ring_capacity_sec = ring_capacity_sec
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.tick_sec = tick_sec
        self._streams: Dict[AudioKey, CircularAudioBuffer] = {}
        self._workers: Dict[AudioKey, SlidingWindowWorker] = {}
        self._lock = asyncio.Lock()
        self._task_group_cm: Optional[anyio.abc.TaskGroup] = None

    async def _ensure_task_group(self):
        if self._task_group_cm is None:
            self._task_group_cm = anyio.create_task_group()
            await self._task_group_cm.__aenter__()

    def set_callback(self, callback: BufferReadyCallback):
        self.callback = callback
        for worker in self._workers.values():
            worker.callback = callback

    async def push_chunk(
        self,
        meeting_id: str,
        participant_id: str,
        track: str,
        wav_data: bytes,
        sample_rate: int,
        channels: int,
    ):
        key: AudioKey = (meeting_id, participant_id, track)
        async with self._lock:
            ring = self._streams.get(key)
            if ring is None:
                ring = CircularAudioBuffer(
                    sample_rate=sample_rate,
                    channels=channels,
                    capacity_sec=self.ring_capacity_sec,
                )
                self._streams[key] = ring
            elif ring.sample_rate != sample_rate:
                logger.warn(
                    "⚠️ [CONTINUOUS_WORKER] Sample rate mismatch for stream; dropping chunk",
                    key=key,
                    expected=ring.sample_rate,
                    received=sample_rate,
                )
                return

            ring.append_wav(wav_data)
            await self.start_if_absent(key)

    async def start_if_absent(self, key: AudioKey):
        if self.callback is None:
            return
        worker = self._workers.get(key)
        if worker is None:
            ring = self._streams[key]
            worker = SlidingWindowWorker(
                key=key,
                ring=ring,
                callback=self.callback,
                window_sec=self.window_sec,
                hop_sec=self.hop_sec,
                tick_sec=self.tick_sec,
            )
            self._workers[key] = worker
            await self._ensure_task_group()
            self._task_group_cm.start_soon(worker.run)

    async def stop_stream(self, key: AudioKey):
        async with self._lock:
            worker = self._workers.pop(key, None)
            self._streams.pop(key, None)
        if worker:
            worker.request_stop()
            await worker.wait_stopped()

    async def stop_all(self):
        async with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
            self._streams.clear()
            task_group = self._task_group_cm
            self._task_group_cm = None
        for worker in workers:
            worker.request_stop()
        for worker in workers:
            await worker.wait_stopped()
        if task_group is not None:
            # Avoid hanging shutdown if a worker callback is stuck.
            task_group.cancel_scope.cancel()
            await task_group.__aexit__(None, None, None)

