from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional

BufferReadyCallback = Callable[..., Awaitable[None]]


class AudioBufferFacade(ABC):
    @abstractmethod
    def set_callback(self, callback: BufferReadyCallback) -> None:
        """Register callback for ready audio payloads."""

    @abstractmethod
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
        """Ingest one audio chunk and optionally return a combined WAV payload."""

    @abstractmethod
    async def stop_stream(self, meeting_id: str, participant_id: str, track: str) -> None:
        """Stop one audio stream and release associated resources."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Stop all streams and release all resources."""

