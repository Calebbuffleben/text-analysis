"""
Orchestrator (spec: modular-feedback-signal-system).
Manages BERT/SBERT lifecycle, threading, and result aggregation.
Pipeline semântico em analysis/semantic_pipeline.run(); dispatch via SignalRegistry.run_all(); merge por signal.key.
No if/else for specific metrics (§3.3). §7: pipeline extraído; linha-count ainda inclui model-loading (poderia extrair loader).
"""

from typing import Dict, Any

from ....types.messages import TranscriptionChunk
from ....models.bert_analyzer import BERTAnalyzer
from ....models.conversation_context import ConversationContext
from ....services.cache_service import AnalysisCache
from ..signals.signal_registry import SignalRegistry
from ....metrics.semantic_metrics import SemanticMetrics
from .semantic_pipeline import run as run_semantic_pipeline
from ....config import Config
import structlog
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = structlog.get_logger()


class TextAnalysisService:
    
    def __init__(self, context_window_size: int = 10, context_window_duration_ms: int = 60000):
        """
        Inicializa serviço de análise.
        
        Args:
        =====
        context_window_size: int, opcional (padrão: 10)
            Número máximo de chunks a manter na janela de contexto
        
        context_window_duration_ms: int, opcional (padrão: 60000)
            Duração da janela de contexto em milissegundos (padrão: 60 segundos)
        """
        self.analyzer = None
        self.cache = AnalysisCache(
            ttl_seconds=Config.CACHE_TTL_SECONDS,
            max_size=Config.CACHE_MAX_SIZE
        )
        
        # Contexto conversacional por participante/reunião
        # Chave: "meetingId:participantId"
        self.conversation_contexts: Dict[str, ConversationContext] = {}
        self.context_window_size = context_window_size
        self.context_window_duration_ms = context_window_duration_ms
        
        # Coletor de métricas semânticas
        self.metrics = SemanticMetrics(alpha=0.1)
        
        # Carregamento de modelos é bloqueante (~60s total), então executamos em thread separada
        # para não bloquear o event loop do asyncio (permite que Socket.IO aceite conexões)
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        # Evita race condition quando múltiplas análises simultâneas tentam carregar modelos
        self._sbert_loading = False

        # Inicializado como None e criado quando necessário (dentro de contexto async)
        try:
            self._model_loading_lock = asyncio.Lock()
        except RuntimeError:
            # Se não houver event loop no momento da inicialização, criar None e inicializar depois
            self._model_loading_lock = None

        self.registry = SignalRegistry()

        logger.info(
            "✅ [SERVIÇO] TextAnalysisService inicializado",
            cache_ttl=Config.CACHE_TTL_SECONDS,
            cache_max_size=Config.CACHE_MAX_SIZE,
            context_window_size=context_window_size,
            context_window_duration_ms=context_window_duration_ms
        )
    
    def _get_analyzer(self) -> BERTAnalyzer:
        """
        Retorna analisador (lazy loading). Usa apenas SBERT para análise semântica.
        
        Returns:
            Instância de BERTAnalyzer
        """
        if self.analyzer is None:
            logger.info("Initializing analyzer (SBERT)")
            self.analyzer = BERTAnalyzer(
                sbert_model_name=getattr(Config, 'SBERT_MODEL_NAME', None)
            )
        return self.analyzer
    
    async def _ensure_models_loaded(self, require_sbert: bool = False):
        """
        Garante que o modelo SBERT está carregado (lazy loading assíncrono).
        
        Executa carregamento no executor para não bloquear event loop.
        Usa lock para evitar carregamento duplicado simultâneo.
        
        Args:
            require_sbert: Se True, carrega SBERT (necessário para análise semântica)
        """
        # Inicializar lock se ainda não foi criado (caso não havia event loop no __init__)
        if self._model_loading_lock is None:
            try:
                self._model_loading_lock = asyncio.Lock()
            except RuntimeError:
                # Se ainda não houver event loop, criar um novo
                # Isso pode acontecer em alguns contextos de teste
                self._model_loading_lock = asyncio.Lock()
        
        # Adquirir lock para evitar carregamento duplicado simultâneo
        async with self._model_loading_lock:
            # Garantir que analyzer existe (instanciar se necessário)
            if self.analyzer is None:
                self._get_analyzer()
            
            # Carregar SBERT se necessário
            if require_sbert and not self.analyzer._sbert_loaded:
                if not self._sbert_loading:
                    self._sbert_loading = True
                    try:
                        logger.info(
                            "🔄 [ANÁLISE] Carregando modelo SBERT em thread separada (não bloqueia event loop)",
                            model=getattr(Config, 'SBERT_MODEL_NAME', None)
                        )
                        
                        # Obter event loop para executar no executor
                        try:
                            loop = asyncio.get_running_loop()
                        except RuntimeError:
                            loop = asyncio.get_event_loop()
                        
                        # Carregar no executor (não bloqueia event loop)
                        # Isso libera o event loop para processar outros eventos (ex: health_ping, conexões Socket.IO)
                        await loop.run_in_executor(
                            self._executor,
                            self._load_sbert_model_sync
                        )
                        
                        logger.info(
                            "✅ [ANÁLISE] Modelo SBERT carregado com sucesso",
                            model=getattr(Config, 'SBERT_MODEL_NAME', None)
                        )
                    finally:
                        self._sbert_loading = False
    
    def _load_sbert_model_sync(self):
        """
        Carrega modelo SBERT de forma síncrona (executar no executor).
        
        Este método é projetado para ser executado em ThreadPoolExecutor,
        permitindo que o carregamento bloqueante (~25-30s) não bloqueie o event loop.
        
        NOTA: Não deve ser chamado diretamente - use _ensure_models_loaded().
        """
        if not self.analyzer:
            self._get_analyzer()
        if not self.analyzer._sbert_loaded:
            self.analyzer._load_sbert_model()
    
    def _get_context_key(self, chunk: TranscriptionChunk) -> str:
        """
        Gera chave única para contexto conversacional.
        
        A chave combina meetingId e participantId para manter contexto
        separado por participante em cada reunião.
        
        Args:
        =====
        chunk: TranscriptionChunk
            Chunk de transcrição
        
        Returns:
        ========
        str: Chave no formato "meetingId:participantId"
        """
        return f"{chunk.meetingId}:{chunk.participantId}"
    
    async def analyze(self, chunk: TranscriptionChunk) -> Dict[str, Any]:
        """
        Analisa texto e retorna resultados completos.
        
        Fluxo:
        1. Verifica cache
        2. Se não encontrado, executa análise
        3. Armazena no cache
        4. Retorna resultados
        
        Args:
            chunk: Chunk de transcrição a ser analisado
            
        Returns:
            Dict com resultados da análise:
            {
                'intent': str,
                'intent_confidence': float,
                'topic': str,
                'topic_confidence': float,
                'speech_act': str,
                'speech_act_confidence': float,
                'keywords': List[str],
                'entities': List[str],
                'sentiment': str,
                'sentiment_score': float,
                'urgency': float,
                'embedding': List[float],
                'sales_category': Optional[str],  # Categoria de vendas detectada (ex: 'price_interest')
                'sales_category_confidence': Optional[float]  # Confiança da classificação (0.0 a 1.0)
            }
        """
        start_time = time.perf_counter()
        
        await self._ensure_models_loaded(require_sbert=True)
        analyzer = self._get_analyzer()
        base, semantic_result = await run_semantic_pipeline(chunk, analyzer, self)
        context = {
            "semantic_result": semantic_result,
            "sbert_enabled": bool(Config.SBERT_MODEL_NAME),
            "meeting_id": chunk.meetingId,
        }
        signal_results = await self.registry.run_all(chunk.text, analyzer, context)
        
        sales_category_flags = semantic_result["sales_category_flags"]
        reformulation_out = signal_results.get("reformulation") or {}
        ref_meta = reformulation_out.get("metadata") or {}
        reformulation_markers_detected = ref_meta.get("reformulation_markers_detected", [])
        reformulation_marker_score = ref_meta.get("reformulation_marker_score", 0.0)
        indecision_out = signal_results.get("indecision_metrics") or {}
        indecision_metrics = indecision_out.get("metadata") if isinstance(indecision_out.get("metadata"), dict) else None

        word_count = base.pop("_word_count", 0)
        char_count = base.pop("_char_count", 0)
        result = {
            **base,
            "sales_category_flags": sales_category_flags,
            "indecision_metrics": indecision_metrics if indecision_metrics else None,
            "reformulation_markers_detected": reformulation_markers_detected,
            "reformulation_marker_score": reformulation_marker_score,
        }
        
        
        # ========================================================================
        # REGISTRO DE MÉTRICAS SEMÂNTICAS
        # ========================================================================
        # Registra métricas de qualidade para monitoramento e ajustes contínuos
        # ========================================================================
        try:
            self.metrics.record_classification(
                category=result.get("sales_category"),
                confidence=result.get("sales_category_confidence") or 0.0,
                intensity=result.get("sales_category_intensity") or 0.0,
                ambiguity=result.get("sales_category_ambiguity") or 1.0,
                flags=result.get("sales_category_flags") or {},
                transition=result.get("sales_category_transition")
            )
        except Exception as e:
            # Não bloquear análise se registro de métricas falhar
            logger.warn(
                "⚠️ [MÉTRICAS] Falha ao registrar métricas",
                error=str(e),
                error_type=type(e).__name__,
                meeting_id=chunk.meetingId
            )

        self.cache.set(
            chunk.meetingId,
            chunk.participantId,
            chunk.text,
            result
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return result
