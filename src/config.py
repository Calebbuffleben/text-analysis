import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """
    Configurações centralizadas do serviço de análise de texto.
    Todas as configurações são carregadas de variáveis de ambiente
    com valores padrão sensatos.
    """
    
    # Server Configuration
    PORT: int = int(os.getenv('PORT', '8000'))
    HOST: str = os.getenv('HOST', '0.0.0.0')
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # Socket.IO Configuration
    SOCKETIO_CORS_ORIGINS: list = os.getenv('SOCKETIO_CORS_ORIGINS', '*').split(',')
    
    # ML Model Configuration
    MODEL_NAME: str = os.getenv('MODEL_NAME', 'neuralmind/bert-base-portuguese-cased')
    MODEL_CACHE_DIR: str = os.getenv('MODEL_CACHE_DIR', '/app/models/.cache')
    MODEL_DEVICE: str = os.getenv('MODEL_DEVICE', 'cpu')
    
    # SBERT Configuration (para análise semântica)
    SBERT_MODEL_NAME: str = os.getenv('SBERT_MODEL_NAME', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # Cache Configuration
    CACHE_TTL_SECONDS: int = int(os.getenv('CACHE_TTL_SECONDS', '300'))
    CACHE_MAX_SIZE: int = int(os.getenv('CACHE_MAX_SIZE', '1000'))
    
    # Performance Configuration
    ANALYSIS_MAX_LENGTH: int = int(os.getenv('ANALYSIS_MAX_LENGTH', '512'))
    ANALYSIS_BATCH_SIZE: int = int(os.getenv('ANALYSIS_BATCH_SIZE', '1'))
    
    # Whisper Configuration (para transcrição de áudio)
    # IMPORTANTE: O default é 'tiny' para melhor performance
    # Se WHISPER_MODEL_NAME não estiver definido, usa 'tiny'
    _whisper_model_env = os.getenv('WHISPER_MODEL_NAME')
    WHISPER_MODEL_NAME: str = _whisper_model_env if _whisper_model_env is not None else 'tiny'
    WHISPER_DEVICE: str = os.getenv('WHISPER_DEVICE', 'cpu')
    WHISPER_LANGUAGE: str = os.getenv('WHISPER_LANGUAGE', 'pt')
    WHISPER_TASK: str = os.getenv('WHISPER_TASK', 'transcribe')
    
    @classmethod
    def validate(cls):
        """Valida configurações críticas"""
        assert cls.MODEL_NAME, "MODEL_NAME must be set"
        assert cls.MODEL_CACHE_DIR, "MODEL_CACHE_DIR must be set"
        assert cls.CACHE_TTL_SECONDS > 0, "CACHE_TTL_SECONDS must be positive"
        assert cls.CACHE_MAX_SIZE > 0, "CACHE_MAX_SIZE must be positive"

