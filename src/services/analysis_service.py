"""
Serviço principal de análise de texto.
Orquestra análise com BERT, gerencia cache e agrega resultados.
"""

from typing import Dict, Any, Tuple, List
from ..types.messages import TranscriptionChunk
from ..models.bert_analyzer import BERTAnalyzer
from ..models.conversation_context import ConversationContext
from ..services.cache_service import AnalysisCache
from ..metrics.semantic_metrics import SemanticMetrics
from ..config import Config
from ..signals.reformulation import (
    detect_reformulation_markers,
    compute_reformulation_marker_score,
    apply_solution_reformulation_signal_flag
)
from ..signals.indecision import compute_indecision_metrics_safe
import structlog
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = structlog.get_logger()


class TextAnalysisService:
    """
    Serviço de análise de texto com BERT.
    
    Responsabilidades:
    - Gerenciar cache de resultados
    - Lazy loading do analisador BERT
    - Orquestrar análise (sentimento, keywords, emoções)
    - Classificação de categorias de vendas usando SBERT
    - Agregar resultados
    
    A classificação de categorias de vendas identifica o estágio da conversa
    usando análise semântica com SBERT. As categorias incluem: price_interest,
    value_exploration, objection_soft, objection_hard, decision_signal,
    information_gathering, stalling, closing_readiness.
    
    O serviço também mantém contexto conversacional para análise temporal,
    permitindo agregação de categorias, detecção de transições e cálculo
    de tendências semânticas ao longo da conversa.
    """
    
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
        
        # FASE 1: ThreadPoolExecutor para carregamento assíncrono de modelos BERT/SBERT
        # Carregamento de modelos é bloqueante (~60s total), então executamos em thread separada
        # para não bloquear o event loop do asyncio (permite que Socket.IO aceite conexões)
        self._executor = ThreadPoolExecutor(max_workers=1)
        
        # FASE 1: Flags para evitar carregamento duplicado de modelos
        # Evita race condition quando múltiplas análises simultâneas tentam carregar modelos
        self._bert_loading = False
        self._sbert_loading = False
        
        # FASE 1: Lock assíncrono para garantir carregamento único de modelos
        # Inicializado como None e criado quando necessário (dentro de contexto async)
        try:
            self._model_loading_lock = asyncio.Lock()
        except RuntimeError:
            # Se não houver event loop no momento da inicialização, criar None e inicializar depois
            self._model_loading_lock = None
        
        logger.info(
            "✅ [SERVIÇO] TextAnalysisService inicializado",
            cache_ttl=Config.CACHE_TTL_SECONDS,
            cache_max_size=Config.CACHE_MAX_SIZE,
            context_window_size=context_window_size,
            context_window_duration_ms=context_window_duration_ms
        )
    
    def _get_analyzer(self) -> BERTAnalyzer:
        """
        Retorna analisador BERT (lazy loading).
        
        Returns:
            Instância de BERTAnalyzer
        """
        if self.analyzer is None:
            logger.info("Initializing BERT analyzer")
            self.analyzer = BERTAnalyzer(
                model_name=Config.MODEL_NAME,
                device=Config.MODEL_DEVICE,
                cache_dir=Config.MODEL_CACHE_DIR,
                max_length=Config.ANALYSIS_MAX_LENGTH,
                sbert_model_name=getattr(Config, 'SBERT_MODEL_NAME', None)
            )
        return self.analyzer
    
    async def _ensure_models_loaded(self, require_sbert: bool = False):
        """
        FASE 2: Garante que modelos BERT/SBERT estão carregados (lazy loading assíncrono).
        
        Executa carregamento no executor para não bloquear event loop.
        Usa lock para evitar carregamento duplicado simultâneo.
        
        Args:
            require_sbert: Se True, também carrega SBERT (necessário para análise semântica)
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
            
            # Carregar BERT se necessário
            if not self.analyzer._loaded:
                if not self._bert_loading:
                    self._bert_loading = True
                    try:
                        logger.info(
                            "🔄 [ANÁLISE] Carregando modelo BERT em thread separada (não bloqueia event loop)",
                            model=Config.MODEL_NAME
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
                            self._load_bert_model_sync
                        )
                        
                        logger.info(
                            "✅ [ANÁLISE] Modelo BERT carregado com sucesso",
                            model=Config.MODEL_NAME
                        )
                    finally:
                        self._bert_loading = False
            
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
    
    def _load_bert_model_sync(self):
        """
        FASE 2: Carrega modelo BERT de forma síncrona (executar no executor).
        
        Este método é projetado para ser executado em ThreadPoolExecutor,
        permitindo que o carregamento bloqueante (~25-35s) não bloqueie o event loop.
        
        NOTA: Não deve ser chamado diretamente - use _ensure_models_loaded().
        """
        if not self.analyzer:
            self._get_analyzer()
        if not self.analyzer._loaded:
            self.analyzer._load_model()
    
    def _load_sbert_model_sync(self):
        """
        FASE 2: Carrega modelo SBERT de forma síncrona (executar no executor).
        
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
        
        logger.debug(
            "🔍 [ANÁLISE] Verificando cache",
            meeting_id=chunk.meetingId,
            participant_id=chunk.participantId,
            text_length=len(chunk.text)
        )
        
        # Verificar cache primeiro
        cached_result = self.cache.get(
            chunk.meetingId,
            chunk.participantId,
            chunk.text
        )
        
        if cached_result:
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "✅ [ANÁLISE] Resultado encontrado no cache",
                meeting_id=chunk.meetingId,
                participant_id=chunk.participantId,
                latency_ms=round(latency_ms, 2)
            )
            return cached_result
        
        logger.info(
            "⚙️ [ANÁLISE] Cache miss, executando análise completa",
            meeting_id=chunk.meetingId,
            participant_id=chunk.participantId,
            text_length=len(chunk.text),
            word_count=len(chunk.text.split())
        )
        
        # FASE 3: Garantir que modelo BERT está carregado (assíncrono, não bloqueia event loop)
        # Isso libera o event loop durante carregamento inicial (~25-35s na primeira chamada)
        # Após carregamento, análises são rápidas (~10-50ms)
        # O carregamento acontece no executor, permitindo que Socket.IO aceite conexões
        await self._ensure_models_loaded(require_sbert=False)
        
        # Obter analisador (agora garantimos que BERT está carregado)
        analyzer = self._get_analyzer()
        
        # FASE 3: Executar análises (modelos já carregados, chamadas são rápidas)
        # Todas essas chamadas são síncronas mas rápidas após carregamento inicial
        logger.debug(
            "📊 [ANÁLISE] Executando análise de sentimento",
            meeting_id=chunk.meetingId
        )
        sentiment = analyzer.analyze_sentiment(chunk.text)
        
        logger.debug(
            "🔑 [ANÁLISE] Extraindo keywords",
            meeting_id=chunk.meetingId
        )
        keywords = analyzer.extract_keywords(chunk.text, top_n=10)
        
        logger.debug(
            "😊 [ANÁLISE] Detectando emoções",
            meeting_id=chunk.meetingId
        )
        emotions = analyzer.detect_emotions(chunk.text)
        
        # Análise semântica com SBERT
        # Esta análise gera embeddings semânticos e pode calcular similaridade
        # com textos anteriores (útil para detectar repetição de ideias)
        semantic_analysis = None
        try:
            if Config.SBERT_MODEL_NAME:
                # FASE 3: Garantir que modelo SBERT está carregado antes de usar
                # Carregamento acontece no executor, não bloqueia event loop
                await self._ensure_models_loaded(require_sbert=True)
                
                logger.debug(
                    "🧠 [ANÁLISE] Executando análise semântica com SBERT",
                    meeting_id=chunk.meetingId
                )
                # Realizar análise semântica completa (SBERT já carregado, chamada é rápida ~50ms)
                # Por enquanto, não passamos textos de referência, mas isso pode ser
                # implementado no futuro para detectar repetição de ideias
                semantic_analysis = analyzer.analyze_semantics(chunk.text)
                logger.debug(
                    "✅ [ANÁLISE] Análise semântica concluída",
                    meeting_id=chunk.meetingId,
                    embedding_dim=semantic_analysis.get('embedding_dimension', 0)
                )
        except Exception as e:
            # Se a análise semântica falhar, continuar sem ela
            logger.warn(
                "⚠️ [ANÁLISE] Análise semântica falhou, continuando sem ela",
                error=str(e),
                meeting_id=chunk.meetingId
            )
        
        # Calcular métricas básicas
        word_count = len(chunk.text.split())
        char_count = len(chunk.text)
        has_question = '?' in chunk.text
        has_exclamation = '!' in chunk.text
        
        logger.debug(
            "📏 [ANÁLISE] Métricas básicas calculadas",
            meeting_id=chunk.meetingId,
            word_count=word_count,
            char_count=char_count,
            has_question=has_question,
            has_exclamation=has_exclamation
        )
        
        # Determinar sentimento como string (maior score)
        sentiment_label = 'neutral'
        sentiment_single_score = sentiment.get('neutral', 0.0)
        if sentiment.get('positive', 0.0) > sentiment.get('negative', 0.0) and sentiment.get('positive', 0.0) > sentiment.get('neutral', 0.0):
            sentiment_label = 'positive'
            sentiment_single_score = sentiment.get('positive', 0.0)
        elif sentiment.get('negative', 0.0) > sentiment.get('neutral', 0.0):
            sentiment_label = 'negative'
            sentiment_single_score = sentiment.get('negative', 0.0)
        
        logger.debug(
            "💭 [ANÁLISE] Sentimento determinado",
            meeting_id=chunk.meetingId,
            sentiment=sentiment_label,
            score=round(sentiment_single_score, 3)
        )
        
        # Detectar intent (intenção) - implementação básica
        logger.debug(
            "🎯 [ANÁLISE] Detectando intenção",
            meeting_id=chunk.meetingId
        )
        intent, intent_confidence = self._detect_intent(chunk.text, has_question)
        
        # Detectar topic (tópico) - implementação básica
        logger.debug(
            "📌 [ANÁLISE] Detectando tópico",
            meeting_id=chunk.meetingId
        )
        topic, topic_confidence = self._detect_topic(chunk.text, keywords)
        
        # Detectar speech_act (ato de fala) - implementação básica
        logger.debug(
            "🗣️ [ANÁLISE] Detectando ato de fala",
            meeting_id=chunk.meetingId
        )
        speech_act, speech_act_confidence = self._detect_speech_act(chunk.text, has_question, has_exclamation)
        
        # Extrair entities (entidades) - implementação básica
        logger.debug(
            "🏷️ [ANÁLISE] Extraindo entidades",
            meeting_id=chunk.meetingId
        )
        entities = self._extract_entities(chunk.text, keywords)
        
        # Calcular urgency (urgência) - implementação básica
        logger.debug(
            "⚡ [ANÁLISE] Calculando urgência",
            meeting_id=chunk.meetingId
        )
        urgency = self._calculate_urgency(sentiment_single_score, has_question, has_exclamation, emotions)
        
        # ========================================================================
        # FASE 9: DETECÇÃO DE KEYWORDS CONDICIONAIS
        # ========================================================================
        # Detecta keywords condicionais que indicam linguagem condicional ou hesitação,
        # característica de clientes indecisos. Usa lista expandida de keywords condicionais.
        # ========================================================================
        conditional_keywords_detected: List[str] = []
        try:
            if Config.SBERT_MODEL_NAME:
                logger.debug(
                    "🔍 [ANÁLISE] Detectando keywords condicionais",
                    meeting_id=chunk.meetingId,
                    text_preview=chunk.text[:50]
                )
                conditional_keywords_detected = analyzer.detect_conditional_keywords(
                    chunk.text,
                    keywords
                )
                if conditional_keywords_detected:
                    logger.debug(
                        "✅ [ANÁLISE] Keywords condicionais detectadas",
                        meeting_id=chunk.meetingId,
                        conditional_keywords=conditional_keywords_detected,
                        count=len(conditional_keywords_detected)
                    )
        except Exception as e:
            # Não bloquear análise se detecção de keywords condicionais falhar
            logger.warn(
                "⚠️ [ANÁLISE] Falha ao detectar keywords condicionais, continuando sem elas",
                error=str(e),
                error_type=type(e).__name__,
                meeting_id=chunk.meetingId
            )
        
        # Obter embedding completo se disponível
        embedding = []
        try:
            if Config.SBERT_MODEL_NAME:
                logger.debug(
                    "🔢 [ANÁLISE] Gerando embedding semântico",
                    meeting_id=chunk.meetingId
                )
                # Gerar embedding completo usando SBERT
                embedding_array = analyzer.generate_semantic_embedding(chunk.text)
                # Converter numpy array para lista Python
                import numpy as np
                if isinstance(embedding_array, np.ndarray):
                    embedding = embedding_array.tolist()
                else:
                    embedding = list(embedding_array)
                logger.debug(
                    "✅ [ANÁLISE] Embedding gerado",
                    meeting_id=chunk.meetingId,
                    embedding_dim=len(embedding)
                )
        except Exception as e:
            logger.warn(
                "⚠️ [ANÁLISE] Falha ao gerar embedding",
                error=str(e),
                meeting_id=chunk.meetingId
            )
            embedding = []
        
        # ========================================================================
        # CLASSIFICAÇÃO DE CATEGORIAS DE VENDAS COM SBERT
        # ========================================================================
        # 
        # Classifica o texto em uma das 8 categorias de vendas usando análise
        # semântica com SBERT. Esta classificação é útil para identificar o estágio
        # da conversa de vendas e fornecer feedback contextualizado.
        #
        # Categorias possíveis:
        # - price_interest: Cliente demonstra interesse em saber o preço
        # - value_exploration: Cliente explora o valor e benefícios da solução
        # - objection_soft: Objeções leves, dúvidas ou hesitações
        # - objection_hard: Objeções fortes e definitivas, rejeição clara
        # - decision_signal: Sinais claros de que o cliente está pronto para decidir
        # - information_gathering: Cliente busca informações adicionais
        # - stalling: Cliente está protelando ou adiando a decisão
        # - closing_readiness: Cliente demonstra prontidão para fechar o negócio
        #
        # A classificação usa exemplos de referência pré-definidos e compara
        # semanticamente o texto com esses exemplos usando embeddings SBERT.
        #
        # Tratamento de erros:
        # - Se SBERT não estiver configurado, sales_category será None
        # - Se a classificação falhar, continua sem ela (não bloqueia outras análises)
        # - Se nenhuma categoria atingir o threshold mínimo, retorna None
        # ========================================================================
        sales_category = None
        sales_category_confidence = None
        sales_category_ambiguity = None
        sales_category_intensity = None
        sales_category_flags: Dict[str, bool] = {}
        try:
            if Config.SBERT_MODEL_NAME:
                logger.debug(
                    "💼 [ANÁLISE] Classificando categoria de vendas com SBERT",
                    meeting_id=chunk.meetingId,
                    text_preview=chunk.text[:50]
                )
                
                # FASE 3: Garantir que modelo SBERT está carregado antes de classificar
                # classify_sales_category() requer SBERT, então precisamos carregar antes
                # Carregamento acontece no executor, não bloqueia event loop
                await self._ensure_models_loaded(require_sbert=True)
                
                # Classificar texto em categoria de vendas
                # O método classify_sales_category() retorna:
                # - categoria: str ou None (nome da categoria detectada)
                # - confiança: float (0.0 a 1.0, score de confiança)
                # - scores: Dict[str, float] (scores de todas as categorias, útil para debugging)
                # - ambiguidade: float (0.0 a 1.0, quão ambíguo é o texto)
                # - intensidade: float (0.0 a 1.0, score absoluto da melhor categoria)
                # - flags: Dict[str, bool] (flags semânticas booleanas)
                categoria, confianca, scores, ambiguidade, intensidade, flags = analyzer.classify_sales_category(
                    chunk.text,
                    min_confidence=0.15  # Threshold mínimo de confiança (15%) - reduzido de 0.3 para permitir mais classificações
                )
                
                # Armazenar resultados
                sales_category = categoria
                sales_category_confidence = confianca
                sales_category_ambiguity = ambiguidade
                sales_category_intensity = intensidade
                sales_category_flags = flags
                
                if sales_category:
                    # Construir reasoning detalhado para logs explicáveis
                    reasoning = {
                        "why": f"High confidence {sales_category} classification",
                        "confidence_reason": f"Large gap between best ({round(scores.get(sales_category, 0.0), 2)}) and second best category",
                        "intensity_reason": f"Absolute score of {round(intensidade, 2)}",
                        "ambiguity_reason": f"Low ambiguity ({round(ambiguidade, 2)})" if ambiguidade < 0.3 else f"Moderate ambiguity ({round(ambiguidade, 2)})"
                    }
                    
                    # Adicionar flags ativas ao reasoning
                    active_flags = [flag for flag, value in flags.items() if value]
                    if active_flags:
                        reasoning["active_flags"] = active_flags
                        reasoning["flags_reason"] = f"Semantic flags triggered: {', '.join(active_flags)}"
                    
                    logger.info(
                        "✅ [ANÁLISE] Categoria de vendas classificada",
                        meeting_id=chunk.meetingId,
                        participant_id=chunk.participantId,
                        sales_category=sales_category,
                        sales_category_confidence=round(confianca, 4),
                        sales_category_intensity=round(intensidade, 4),
                        sales_category_ambiguity=round(ambiguidade, 4),
                        sales_category_flags=flags,
                        best_score=round(scores.get(sales_category, 0.0), 4) if scores else 0.0,
                        reasoning=reasoning
                    )
                else:
                    # Construir reasoning para caso sem categoria detectada
                    best_score = max(scores.values()) if scores else 0.0
                    reasoning = {
                        "why": "No category met minimum confidence threshold",
                        "reason": f"Best score {round(best_score, 2)} < {0.15}",
                        "ambiguity_reason": f"Ambiguity: {round(ambiguidade, 2)}" if ambiguidade else "N/A"
                    }
                    
                    logger.debug(
                        "⚠️ [ANÁLISE] Nenhuma categoria de vendas detectada com confiança suficiente",
                        meeting_id=chunk.meetingId,
                        best_score=round(best_score, 4),
                        ambiguity=round(ambiguidade, 4) if ambiguidade else None,
                        intensity=round(intensidade, 4) if intensidade else None,
                        min_confidence=0.15,
                        reasoning=reasoning
                    )
        except Exception as e:
            # Se a classificação de categoria de vendas falhar, continuar sem ela
            # Isso não deve bloquear outras análises
            logger.warn(
                "⚠️ [ANÁLISE] Falha ao classificar categoria de vendas, continuando sem ela",
                error=str(e),
                error_type=type(e).__name__,
                meeting_id=chunk.meetingId
            )
            sales_category = None
            sales_category_confidence = None
            sales_category_ambiguity = None
            sales_category_intensity = None
            sales_category_flags = {}

        # ========================================================================
        # (Opcional) Reformulação do cliente ("solução foi compreendida")
        # ========================================================================
        # Detecta marcadores linguísticos de reformulação/teach-back no texto atual.
        # OBS: não tenta calcular similaridade com contexto (isso é melhor no backend,
        # que já gerencia estado por meeting e cooldowns).
        reformulation_markers_detected = detect_reformulation_markers(chunk.text)
        reformulation_marker_score = compute_reformulation_marker_score(reformulation_markers_detected)
        apply_solution_reformulation_signal_flag(sales_category_flags, reformulation_marker_score)
        
        # ========================================================================
        # FASE 10: CÁLCULO DE MÉTRICAS DE INDECISÃO
        # ========================================================================
        # Calcula métricas específicas de indecisão para facilitar análise no backend.
        # Métricas pré-calculadas reduzem processamento no backend e podem ser
        # usadas em múltiplas heurísticas.
        # IMPORTANTE: Deve vir APÓS a classificação de categoria de vendas.
        # ========================================================================
        sbert_enabled = bool(Config.SBERT_MODEL_NAME)
        indecision_metrics = compute_indecision_metrics_safe(
            analyzer=analyzer,
            sbert_enabled=sbert_enabled,
            sales_category=sales_category,
            sales_category_confidence=sales_category_confidence,
            sales_category_intensity=sales_category_intensity,
            sales_category_ambiguity=sales_category_ambiguity,
            conditional_keywords_detected=conditional_keywords_detected,
            meeting_id=chunk.meetingId
        )
        if indecision_metrics:
            logger.debug(
                "✅ [ANÁLISE] Métricas de indecisão calculadas",
                meeting_id=chunk.meetingId,
                indecision_score=round(indecision_metrics.get('indecision_score', 0.0), 4),
                postponement_likelihood=round(indecision_metrics.get('postponement_likelihood', 0.0), 4),
                conditional_language_score=round(indecision_metrics.get('conditional_language_score', 0.0), 4)
            )
        
        # ========================================================================
        # CONTEXTO CONVERSACIONAL
        # ========================================================================
        # Adicionar chunk ao contexto conversacional para análise temporal
        # O contexto permite análise de padrões ao longo da conversa, como:
        # - Agregação temporal de categorias
        # - Detecção de transições de estágio
        # - Cálculo de tendências semânticas
        # - Redução de ruído de frases isoladas
        # ========================================================================
        context_key = self._get_context_key(chunk)
        if context_key not in self.conversation_contexts:
            self.conversation_contexts[context_key] = ConversationContext(
                window_size=self.context_window_size,
                window_duration_ms=self.context_window_duration_ms
            )
        
        context = self.conversation_contexts[context_key]
        context.add_chunk({
            'text': chunk.text,
            'sales_category': sales_category,
            'sales_category_confidence': sales_category_confidence,
            'sales_category_intensity': sales_category_intensity,
            'sales_category_ambiguity': sales_category_ambiguity,
            'timestamp': chunk.timestamp,
            'embedding': embedding
        })
        
        # Obter janela de contexto para análise temporal
        window = context.get_window(chunk.timestamp)
        
        # ========================================================================
        # ANÁLISE CONTEXTUAL: Agregação, Transições e Tendências
        # ========================================================================
        # 
        # Usa contexto histórico para análises mais robustas:
        # - Agregação temporal: reduz ruído de frases isoladas
        # - Detecção de transições: identifica mudanças de estágio
        # - Tendência semântica: indica direção da conversa
        # ========================================================================
        sales_category_aggregated = None
        sales_category_transition = None
        sales_category_trend = None
        
        try:
            if window and Config.SBERT_MODEL_NAME:
                analyzer = self._get_analyzer()
                
                # 1. Agregar categorias temporais para reduzir ruído
                sales_category_aggregated = analyzer.aggregate_categories_temporal(window)
                
                if sales_category_aggregated:
                    logger.debug(
                        "📊 [CONTEXTO] Categorias agregadas temporalmente",
                        meeting_id=chunk.meetingId,
                        participant_id=chunk.participantId,
                        dominant_category=sales_category_aggregated['dominant_category'],
                        stability=round(sales_category_aggregated['stability'], 4),
                        distribution=sales_category_aggregated['category_distribution']
                    )
                
                # 2. Detectar transições de categoria
                if sales_category and sales_category_confidence is not None:
                    sales_category_transition = analyzer.detect_category_transition(
                        sales_category,
                        sales_category_confidence,
                        window
                    )
                    
                    if sales_category_transition:
                        logger.info(
                            "🔄 [CONTEXTO] Transição de categoria detectada",
                            meeting_id=chunk.meetingId,
                            participant_id=chunk.participantId,
                            transition_type=sales_category_transition['transition_type'],
                            from_category=sales_category_transition['from_category'],
                            to_category=sales_category_transition['to_category'],
                            confidence=round(sales_category_transition['confidence'], 4),
                            time_delta_ms=sales_category_transition['time_delta_ms']
                        )
                
                # 3. Calcular tendência semântica
                sales_category_trend = analyzer.calculate_semantic_trend(window)
                
                if sales_category_trend:
                    logger.debug(
                        "📈 [CONTEXTO] Tendência semântica calculada",
                        meeting_id=chunk.meetingId,
                        participant_id=chunk.participantId,
                        trend=sales_category_trend['trend'],
                        trend_strength=round(sales_category_trend['trend_strength'], 4),
                        current_stage=sales_category_trend['current_stage'],
                        velocity=round(sales_category_trend['velocity'], 4)
                    )
        except Exception as e:
            logger.warn(
                "⚠️ [CONTEXTO] Falha ao realizar análises contextuais",
                error=str(e),
                error_type=type(e).__name__,
                meeting_id=chunk.meetingId
            )
        
        logger.debug(
            "📊 [CONTEXTO] Chunk adicionado ao contexto conversacional",
            meeting_id=chunk.meetingId,
            participant_id=chunk.participantId,
            context_key=context_key,
            window_size=len(window),
            history_size=context.get_history_size()
        )
        
        # ========================================================================
        # REGISTRO DE MÉTRICAS SEMÂNTICAS
        # ========================================================================
        # Registra métricas de qualidade para monitoramento e ajustes contínuos
        # ========================================================================
        try:
            self.metrics.record_classification(
                category=sales_category,
                confidence=sales_category_confidence or 0.0,
                intensity=sales_category_intensity or 0.0,
                ambiguity=sales_category_ambiguity or 1.0,
                flags=sales_category_flags,
                transition=sales_category_transition
            )
        except Exception as e:
            # Não bloquear análise se registro de métricas falhar
            logger.warn(
                "⚠️ [MÉTRICAS] Falha ao registrar métricas",
                error=str(e),
                error_type=type(e).__name__,
                meeting_id=chunk.meetingId
            )
        
        # Construir resultado completo com nova estrutura
        result = {
            'intent': intent,
            'intent_confidence': intent_confidence,
            'topic': topic,
            'topic_confidence': topic_confidence,
            'speech_act': speech_act,
            'speech_act_confidence': speech_act_confidence,
            'keywords': keywords,
            'entities': entities,
            'sentiment': sentiment_label,
            'sentiment_score': sentiment_single_score,
            'urgency': urgency,
            'embedding': embedding,
            # Categorias de vendas classificadas com SBERT
            'sales_category': sales_category,
            'sales_category_confidence': sales_category_confidence,
            'sales_category_intensity': sales_category_intensity,
            'sales_category_ambiguity': sales_category_ambiguity,
            'sales_category_flags': sales_category_flags,
            # Análises contextuais (baseadas em histórico)
            'sales_category_aggregated': sales_category_aggregated,
            'sales_category_transition': sales_category_transition,
            'sales_category_trend': sales_category_trend,
            # Keywords condicionais detectadas (FASE 9)
            'conditional_keywords_detected': conditional_keywords_detected,
            # Métricas de indecisão (FASE 10)
            'indecision_metrics': indecision_metrics if indecision_metrics else None,
            # Reformulação do cliente (teach-back) — sinais leves (sem contexto)
            'reformulation_markers_detected': reformulation_markers_detected,
            'reformulation_marker_score': reformulation_marker_score
        }
        
        # Armazenar no cache
        logger.debug(
            "💾 [ANÁLISE] Armazenando resultado no cache",
            meeting_id=chunk.meetingId,
            participant_id=chunk.participantId
        )
        self.cache.set(
            chunk.meetingId,
            chunk.participantId,
            chunk.text,
            result
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            "✅ [ANÁLISE] Análise completa concluída",
            meeting_id=chunk.meetingId,
            participant_id=chunk.participantId,
            word_count=word_count,
            char_count=char_count,
            sentiment=sentiment_label,
            sentiment_score=round(sentiment_single_score, 3),
            intent=intent,
            intent_confidence=round(intent_confidence, 3),
            topic=topic,
            topic_confidence=round(topic_confidence, 3),
            speech_act=speech_act,
            speech_act_confidence=round(speech_act_confidence, 3),
            urgency=round(urgency, 3),
            keywords_count=len(keywords),
            entities_count=len(entities),
            embedding_dim=len(embedding),
            sales_category=sales_category,
            sales_category_confidence=round(sales_category_confidence, 4) if sales_category_confidence is not None else None,
            sales_category_intensity=round(sales_category_intensity, 4) if sales_category_intensity is not None else None,
            sales_category_ambiguity=round(sales_category_ambiguity, 4) if sales_category_ambiguity is not None else None,
            sales_category_flags=sales_category_flags if sales_category_flags else None,
            latency_ms=round(latency_ms, 2)
        )
        
        return result
    
    def _detect_intent(self, text: str, has_question: bool) -> Tuple[str, float]:
        """
        Detecta intenção do texto (implementação básica).
        
        Args:
            text: Texto a ser analisado
            has_question: Se contém interrogação
            
        Returns:
            Tupla (intent, confidence)
        """
        text_lower = text.lower()
        
        # Mapeamento básico de intenções
        intent_patterns = {
            'ask_price': ['quanto', 'custa', 'valor', 'preço', 'price'],
            'ask_info': ['o que', 'como', 'quando', 'onde', 'quem'],
            'request_action': ['pode', 'poderia', 'favor', 'por favor', 'faça'],
            'express_opinion': ['acho', 'penso', 'acredito', 'opinião'],
            'express_agreement': ['concordo', 'sim', 'exato', 'certo'],
            'express_disagreement': ['discordo', 'não', 'errado', 'incorreto']
        }
        
        for intent, patterns in intent_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                # Calcular confiança baseada em quantos padrões foram encontrados
                matches = sum(1 for pattern in patterns if pattern in text_lower)
                confidence = min(0.9, 0.5 + (matches * 0.1))
                return (intent, confidence)
        
        # Default: intent genérico
        if has_question:
            return ('ask_question', 0.6)
        return ('statement', 0.5)
    
    def _detect_topic(self, text: str, keywords: List[str]) -> Tuple[str, float]:
        """
        Detecta tópico do texto (implementação básica).
        
        Args:
            text: Texto a ser analisado
            keywords: Lista de keywords extraídas
            
        Returns:
            Tupla (topic, confidence)
        """
        text_lower = text.lower()
        
        # Mapeamento básico de tópicos
        topic_patterns = {
            'pricing': ['preço', 'valor', 'custo', 'price', 'quanto'],
            'product': ['produto', 'serviço', 'solução', 'oferta'],
            'support': ['suporte', 'ajuda', 'problema', 'erro', 'bug'],
            'schedule': ['agendar', 'horário', 'data', 'reunião', 'meeting'],
            'technical': ['técnico', 'implementação', 'código', 'tecnologia']
        }
        
        for topic, patterns in topic_patterns.items():
            if any(pattern in text_lower for pattern in patterns) or any(kw in patterns for kw in keywords):
                matches = sum(1 for pattern in patterns if pattern in text_lower or pattern in keywords)
                confidence = min(0.95, 0.6 + (matches * 0.1))
                return (topic, confidence)
        
        # Default: tópico genérico
        return ('general', 0.5)
    
    def _detect_speech_act(self, text: str, has_question: bool, has_exclamation: bool) -> Tuple[str, float]:
        """
        Detecta ato de fala (speech act) do texto.
        
        Args:
            text: Texto a ser analisado
            has_question: Se contém interrogação
            has_exclamation: Se contém exclamação
            
        Returns:
            Tupla (speech_act, confidence)
        """
        text_lower = text.lower()
        
        if has_question:
            return ('question', 0.9)
        
        if has_exclamation:
            return ('exclamation', 0.85)
        
        # Verificar padrões de comandos
        command_patterns = ['favor', 'por favor', 'pode', 'poderia', 'faça', 'execute']
        if any(pattern in text_lower for pattern in command_patterns):
            return ('request', 0.8)
        
        # Verificar padrões de afirmação
        if any(word in text_lower for word in ['sim', 'certo', 'ok', 'entendi', 'concordo']):
            return ('agreement', 0.75)
        
        if any(word in text_lower for word in ['não', 'discordo', 'errado', 'incorreto']):
            return ('disagreement', 0.75)
        
        # Default: statement
        return ('statement', 0.7)
    
    def _extract_entities(self, text: str, keywords: List[str]) -> List[str]:
        """
        Extrai entidades do texto (implementação básica).
        
        Args:
            text: Texto a ser analisado
            keywords: Lista de keywords extraídas
            
        Returns:
            Lista de entidades encontradas
        """
        text_lower = text.lower()
        entities = []
        
        # Entidades comuns (pode ser expandido com NER)
        entity_patterns = {
            'preço': ['preço', 'valor', 'custo', 'price'],
            'produto': ['produto', 'serviço', 'solução'],
            'data': ['hoje', 'amanhã', 'semana', 'mês', 'ano'],
            'pessoa': ['você', 'eu', 'nós', 'eles']
        }
        
        for entity, patterns in entity_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                entities.append(entity)
        
        # Adicionar keywords relevantes como entidades
        for kw in keywords[:3]:  # Top 3 keywords
            if kw not in entities and len(kw) > 3:
                entities.append(kw)
        
        return entities[:5]  # Limitar a 5 entidades
    
    def _calculate_urgency(self, sentiment_score: float, has_question: bool, has_exclamation: bool, emotions: dict[str, float]) -> float:
        """
        Calcula urgência do texto (0.0 a 1.0).
        
        Args:
            sentiment_score: Score de sentimento
            has_question: Se contém interrogação
            has_exclamation: Se contém exclamação
            emotions: Dict de emoções
            
        Returns:
            Score de urgência (0.0 a 1.0)
        """
        urgency = 0.5  # Base
        
        # Perguntas aumentam urgência
        if has_question:
            urgency += 0.15
        
        # Exclamações aumentam urgência
        if has_exclamation:
            urgency += 0.1
        
        # Emoções negativas aumentam urgência
        negative_emotions = emotions.get('anger', 0.0) + emotions.get('fear', 0.0)
        urgency += negative_emotions * 0.2
        
        # Sentimento negativo aumenta urgência
        if sentiment_score < 0.4:
            urgency += 0.1
        
        return min(1.0, max(0.0, urgency))

