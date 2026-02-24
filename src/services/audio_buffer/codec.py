import io

import numpy as np
import soundfile as sf


def wav_to_pcm_int16(wav_data: bytes) -> np.ndarray:
    """Decode WAV bytes to mono int16 samples."""
    audio_float, _ = sf.read(io.BytesIO(wav_data), dtype="float32")
    if isinstance(audio_float, np.ndarray) and audio_float.ndim == 2:
        audio_float = audio_float.mean(axis=1)
    audio_float = np.clip(audio_float, -1.0, 1.0)
    return (audio_float * 32767.0).astype(np.int16)


def pcm_int16_to_wav(pcm: np.ndarray, sample_rate: int, channels: int = 1) -> bytes:
    """Encode mono int16 samples to WAV bytes."""
    audio_float = pcm.astype(np.float32, copy=False) / 32768.0
    if channels > 1:
        audio_float = np.repeat(audio_float[:, None], channels, axis=1)
    wav_output = io.BytesIO()
    sf.write(wav_output, audio_float, sample_rate, format="WAV", subtype="PCM_16")
    return wav_output.getvalue()

