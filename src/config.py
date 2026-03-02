import json
from typing import Any, List

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Server Configuration
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    PUBLIC_HOSTNAME: str = "backend-analysis-production-a688.up.railway.app"
    LOG_LEVEL: str = "INFO"

    # Socket.IO Configuration
    SOCKETIO_CORS_ORIGINS: str | List[str] | None = Field(default="*")

    # SBERT Configuration
    SBERT_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Cache Configuration
    CACHE_TTL_SECONDS: int = 300
    CACHE_MAX_SIZE: int = 1000

    # Performance Configuration
    ANALYSIS_BATCH_SIZE: int = 1

    # Whisper Configuration
    WHISPER_MODEL_NAME: str = "tiny"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_LANGUAGE: str = "pt"
    WHISPER_TASK: str = "transcribe"

    # Transcription Concurrency
    MAX_CONCURRENT_TRANSCRIPTIONS: int = 2

    # Continuous worker architecture flags
    CONTINUOUS_WORKER_ENABLED: bool = False
    CONTINUOUS_RING_CAPACITY_SEC: float = 20.0
    CONTINUOUS_WINDOW_SEC: float = 5.0
    CONTINUOUS_HOP_SEC: float = 1.0
    CONTINUOUS_TICK_SEC: float = 1.0

    # Dedupe configuration — TTL must cover the full circular buffer lifetime
    # so the same speech is never re-emitted while still in the buffer.
    RESULT_DEDUPE_TTL_SEC: float = 30.0
    RESULT_DEDUPE_MAX_SIZE: int = 20000

    # Rate limiter: minimum interval between result emissions per participant
    RESULT_MIN_INTERVAL_SEC: float = 5.0

    @field_validator("SOCKETIO_CORS_ORIGINS", mode="before")
    @classmethod
    def _split_cors_origins(cls, v: Any) -> List[str]:
        # Robust env normalization for deployment-safe parsing.
        # Supports:
        # - JSON array: ["https://a.com","https://b.com"]
        # - single value: https://example.com
        # - comma-separated: https://a.com,https://b.com
        # - wildcard: *
        # - empty/undefined: defaults to ["*"]
        default = ["*"]

        if v is None:
            return default

        if isinstance(v, list):
            normalized = [str(item).strip() for item in v if str(item).strip()]
            return normalized or default

        if isinstance(v, str):
            raw = v.strip()
            if raw == "":
                return default
            if raw == "*":
                return default

            # JSON list input (preferred in strict env setups)
            if raw.startswith("["):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        normalized = [str(item).strip() for item in parsed if str(item).strip()]
                        return normalized or default
                except (json.JSONDecodeError, TypeError, ValueError):
                    # Fall through to string parsing fallback
                    pass

            # Comma-separated or single value
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            return parts or default

        return default


_settings = _Settings()


class Config:
    """
    Compat layer to preserve Config.X access pattern.
    Values are now loaded and validated via pydantic-settings.
    """

    PORT = _settings.PORT
    HOST = _settings.HOST
    PUBLIC_HOSTNAME = _settings.PUBLIC_HOSTNAME
    LOG_LEVEL = _settings.LOG_LEVEL
    SOCKETIO_CORS_ORIGINS = _settings.SOCKETIO_CORS_ORIGINS
    SBERT_MODEL_NAME = _settings.SBERT_MODEL_NAME
    CACHE_TTL_SECONDS = _settings.CACHE_TTL_SECONDS
    CACHE_MAX_SIZE = _settings.CACHE_MAX_SIZE
    ANALYSIS_BATCH_SIZE = _settings.ANALYSIS_BATCH_SIZE
    WHISPER_MODEL_NAME = _settings.WHISPER_MODEL_NAME
    WHISPER_DEVICE = _settings.WHISPER_DEVICE
    WHISPER_LANGUAGE = _settings.WHISPER_LANGUAGE
    WHISPER_TASK = _settings.WHISPER_TASK
    MAX_CONCURRENT_TRANSCRIPTIONS = _settings.MAX_CONCURRENT_TRANSCRIPTIONS
    CONTINUOUS_WORKER_ENABLED = _settings.CONTINUOUS_WORKER_ENABLED
    CONTINUOUS_RING_CAPACITY_SEC = _settings.CONTINUOUS_RING_CAPACITY_SEC
    CONTINUOUS_WINDOW_SEC = _settings.CONTINUOUS_WINDOW_SEC
    CONTINUOUS_HOP_SEC = _settings.CONTINUOUS_HOP_SEC
    CONTINUOUS_TICK_SEC = _settings.CONTINUOUS_TICK_SEC
    RESULT_DEDUPE_TTL_SEC = _settings.RESULT_DEDUPE_TTL_SEC
    RESULT_DEDUPE_MAX_SIZE = _settings.RESULT_DEDUPE_MAX_SIZE
    RESULT_MIN_INTERVAL_SEC = _settings.RESULT_MIN_INTERVAL_SEC

    @classmethod
    def validate(cls):
        """Validate critical config constraints."""
        try:
            _Settings()
        except ValidationError as exc:
            raise AssertionError(f"Invalid configuration: {exc}") from exc
        assert cls.CACHE_TTL_SECONDS > 0, "CACHE_TTL_SECONDS must be positive"
        assert cls.CACHE_MAX_SIZE > 0, "CACHE_MAX_SIZE must be positive"

