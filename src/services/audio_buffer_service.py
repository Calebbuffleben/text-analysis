"""
Compatibility shim for legacy imports.

Public classes were moved to services/audio_buffer/*.
This file is intentionally minimal to avoid symbol override.
"""

from .audio_buffer import (  # noqa: F401
    AudioBuffer,
    AudioBufferFacade,
    AudioBufferService,
    AudioKey,
    BufferReadyCallback,
    CircularAudioBuffer,
    ContinuousAudioStreamManager,
    SlidingWindowWorker,
)
