import json
from typing import Any, List

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class _Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Server Configuration
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    PUBLIC_HOSTNAME: str = "backend-analysis-production-a688.up.railway.app"
    LOG_LEVEL: str = "INFO"

    # Socket.IO Configuration
    SOCKETIO_CORS_ORIGINS: NoDecode[List[str]] = Field(default_factory=lambda: ["*"])

    # ML Model Configuration
    MODEL_NAME: str = "neuralmind/bert-base-portuguese-cased"
    MODEL_CACHE_DIR: str = "/app/models/.cache"
    MODEL_DEVICE: str = "cpu"

    # SBERT Configuration
    SBERT_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Cache Configuration
    CACHE_TTL_SECONDS: int = 300
    CACHE_MAX_SIZE: int = 1000

    # Performance Configuration
    ANALYSIS_MAX_LENGTH: int = 512
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

    # Dedupe configuration
    RESULT_DEDUPE_TTL_SEC: float = 6.0
    RESULT_DEDUPE_MAX_SIZE: int = 20000

    @field_validator("SOCKETIO_CORS_ORIGINS", mode="before")
    @classmethod
    def _split_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            raw = v.strip()
            if raw == "":
                raise ValueError(
                    "SOCKETIO_CORS_ORIGINS must be a JSON array string, e.g. "
                    '["http://localhost:3000"]'
                )
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "SOCKETIO_CORS_ORIGINS must be valid JSON array syntax, e.g. "
                    '["http://localhost:3000"] or []'
                ) from exc
            if not isinstance(parsed, list):
                raise ValueError(
                    "SOCKETIO_CORS_ORIGINS must be a JSON array, e.g. "
                    '["http://localhost:3000"] or []'
                )
            if not all(isinstance(item, str) for item in parsed):
                raise ValueError("SOCKETIO_CORS_ORIGINS must contain only string origins")
            return parsed
        if isinstance(v, list):
            return v
        return ["*"]


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
    MODEL_NAME = _settings.MODEL_NAME
    MODEL_CACHE_DIR = _settings.MODEL_CACHE_DIR
    MODEL_DEVICE = _settings.MODEL_DEVICE
    SBERT_MODEL_NAME = _settings.SBERT_MODEL_NAME
    CACHE_TTL_SECONDS = _settings.CACHE_TTL_SECONDS
    CACHE_MAX_SIZE = _settings.CACHE_MAX_SIZE
    ANALYSIS_MAX_LENGTH = _settings.ANALYSIS_MAX_LENGTH
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

    @classmethod
    def validate(cls):
        """Validate critical config constraints."""
        try:
            _Settings()
        except ValidationError as exc:
            raise AssertionError(f"Invalid configuration: {exc}") from exc
        assert cls.MODEL_NAME, "MODEL_NAME must be set"
        assert cls.MODEL_CACHE_DIR, "MODEL_CACHE_DIR must be set"
        assert cls.CACHE_TTL_SECONDS > 0, "CACHE_TTL_SECONDS must be positive"
        assert cls.CACHE_MAX_SIZE > 0, "CACHE_MAX_SIZE must be positive"

