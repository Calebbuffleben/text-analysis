"""
Entry point do serviço de análise de texto.
Integra FastAPI (para endpoints REST) com Socket.IO (para comunicação real-time).
"""

import uvicorn
import structlog
import time
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .config import Config
from .socketio_server import app as socketio_app
from .services.analysis_service import TextAnalysisService
from .types.messages import TranscriptionChunk

# Configurar structlog para exibir logs no console do Docker
# Usar ConsoleRenderer para logs legíveis no console
# Se LOG_FORMAT=json, usar JSONRenderer (útil para produção/agregação)
import os

log_format = os.getenv('LOG_FORMAT', 'console').lower()

if log_format == 'json':
    # Formato JSON (útil para produção, logs agregados)
    renderer = structlog.processors.JSONRenderer()
else:
    # Formato console legível (padrão para desenvolvimento/Docker)
    # Usar KeyValueRenderer que é mais compatível e funciona bem no Docker
    try:
        # Tentar usar ConsoleRenderer se disponível (structlog >= 22.0)
        renderer = structlog.dev.ConsoleRenderer(
            colors=False,  # Desabilitar cores no Docker
            exception_formatter=structlog.dev.plain_traceback
        )
    except (AttributeError, ImportError):
        # Fallback para KeyValueRenderer (mais compatível)
        renderer = structlog.processors.KeyValueRenderer(
            key_order=['timestamp', 'level', 'event', 'logger'],
            drop_missing=True
        )

# DIAGNÓSTICO: Configurar nível de log baseado em LOG_LEVEL
import logging
log_level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level)

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,  # Este processor filtra por nível
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        renderer
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# DIAGNÓSTICO: Log de configuração
print(f"[DIAGNÓSTICO] Structlog configurado - LOG_LEVEL={log_level_str}, log_level={log_level}")

logger = structlog.get_logger()

# Log de inicialização do structlog
logger.info(
    "✅ [SISTEMA] Structlog configurado",
    log_format=log_format,
    log_level=os.getenv('LOG_LEVEL', 'INFO')
)

# Criar app FastAPI
fastapi_app = FastAPI(
    title="Text Analysis Service",
    description="Serviço de análise de texto com BERT para português",
    version="1.0.0"
)

# Configurar CORS para permitir requisições HTTP de polling do Socket.IO
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas as origens (ou configurar específicas para produção)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instanciar serviço de análise
logger.info("🔄 [SISTEMA] Criando instâncias de serviços...")
analysis_service = TextAnalysisService()
logger.info("✅ [SISTEMA] Serviços criados com sucesso")


@fastapi_app.get("/health")
async def health():
    """
    Health check endpoint.
    
    Returns:
        JSON com status do serviço
    """
    return JSONResponse({
        "status": "ok",
        "service": "text-analysis",
        "version": "1.0.0"
    })


@fastapi_app.post("/analyze")
async def analyze_text(request: dict):
    """
    Endpoint REST para análise de texto (para testes e debugging).
    
    Request Body:
    {
        "text": "Texto a ser analisado",
        "meetingId": "meet_123",
        "participantId": "user_456",
        "timestamp": 1234567890
    }
    
    Response:
    {
        "meetingId": "meet_123",
        "participantId": "user_456",
        "text": "Texto a ser analisado",
        "analysis": {
            "word_count": 5,
            "char_count": 25,
            "has_question": false,
            "has_exclamation": false,
            "sentiment_score": {
                "positive": 0.7,
                "negative": 0.1,
                "neutral": 0.2
            },
            "emotions": {...},
            "topics": [],
            "keywords": ["texto", "analisado"]
        },
        "timestamp": 1234567890,
        "confidence": 0.9
    }
    """
    try:
        # Validar request
        if not request.get('text'):
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Criar chunk
        chunk = TranscriptionChunk(
            meetingId=request.get('meetingId', 'test_meeting'),
            participantId=request.get('participantId', 'test_user'),
            text=request.get('text', ''),
            timestamp=request.get('timestamp', 0)
        )
        
        # Processar análise
        analysis_result = await analysis_service.analyze(chunk)
        
        # Retornar resultado
        return JSONResponse({
            "meetingId": chunk.meetingId,
            "participantId": chunk.participantId,
            "text": chunk.text,
            "analysis": analysis_result,
            "timestamp": chunk.timestamp,
            "confidence": 0.9
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in /analyze endpoint", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@fastapi_app.get("/cache/stats")
async def cache_stats():
    """
    Endpoint para estatísticas do cache.
    
    Returns:
        JSON com estatísticas do cache
    """
    stats = analysis_service.cache.stats()
    return JSONResponse(stats)


@fastapi_app.get("/metrics")
async def get_metrics():
    """
    Endpoint para métricas semânticas de qualidade.
    
    Retorna métricas agregadas sobre a classificação de categorias de vendas,
    incluindo:
    - Taxa de sucesso de classificações
    - Confiança, intensidade e ambiguidade médias
    - Distribuição de categorias detectadas
    - Taxa de alta confiança
    - Contagem de flags e transições
    
    Returns:
        JSON com métricas semânticas
    """
    try:
        metrics = analysis_service.metrics.get_metrics()
        summary = analysis_service.metrics.get_summary()
        
        return JSONResponse({
            "status": "ok",
            "metrics": metrics,
            "summary": summary,
            "timestamp": time.time()
        })
    except Exception as e:
        logger.error("Error in /metrics endpoint", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@fastapi_app.post("/cache/clear")
async def clear_cache():
    """
    Endpoint para limpar cache (útil para testes).
    
    Returns:
        JSON com confirmação
    """
    analysis_service.cache.clear()
    return JSONResponse({"status": "cache cleared"})


# Montar Socket.IO app no FastAPI
logger.info("🔌 [SISTEMA] Montando Socket.IO no FastAPI...")
fastapi_app.mount("/socket.io/", socketio_app)
logger.info("✅ [SISTEMA] Socket.IO montado com sucesso")

if __name__ == "__main__":
    # Validar configurações
    Config.validate()
    
    logger.info(
        "🚀 [INICIALIZAÇÃO] Iniciando Text Analysis Service",
        host=Config.PUBLIC_HOSTNAME,  # Use public hostname for logs
        bind_host=Config.HOST,  # Show bind host (0.0.0.0) separately
        port=Config.PORT,
        model=Config.MODEL_NAME,
        sbert_model=Config.SBERT_MODEL_NAME,
        whisper_model=Config.WHISPER_MODEL_NAME,
        device=Config.MODEL_DEVICE,
        cache_ttl=Config.CACHE_TTL_SECONDS,
        cache_max_size=Config.CACHE_MAX_SIZE
    )
    
    # Configurar logging do uvicorn para não interferir com structlog
    import logging
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.WARNING)  # Reduzir logs do uvicorn
    
    uvicorn.run(
        fastapi_app,
        host=Config.HOST,
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower(),
        access_log=False,  # Usar structlog ao invés
        use_colors=False,  # Desabilitar cores no Docker
        log_config=None  # Não usar configuração padrão do uvicorn
    )

