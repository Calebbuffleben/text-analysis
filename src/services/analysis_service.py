"""
Serviço principal de análise de texto.
Orquestra análise com BERT, gerencia cache e agrega resultados.
"""

from typing import Dict, Any, Tuple, List
from ..types.messages import TranscriptionChunk
from ..models.bert_analyzer import BERTAnalyzer
from ..services.cache_service import AnalysisCache
from ..config import Config
import structlog
import time

logger = structlog.get_logger()


class TextAnalysisService:
    """
    Serviço de análise de texto com BERT.
    
    Responsabilidades:
    - Gerenciar cache de resultados
    - Lazy loading do analisador BERT
    - Orquestrar análise (sentimento, keywords, emoções)
    - Agregar resultados
    """
    
    def __init__(self):
        """Inicializa serviço de análise"""
        self.analyzer = None
        self.cache = AnalysisCache(
            ttl_seconds=Config.CACHE_TTL_SECONDS,
            max_size=Config.CACHE_MAX_SIZE
        )
        
        logger.info(
            "✅ [SERVIÇO] TextAnalysisService inicializado",
            cache_ttl=Config.CACHE_TTL_SECONDS,
            cache_max_size=Config.CACHE_MAX_SIZE
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
                'word_count': int,
                'char_count': int,
                'has_question': bool,
                'has_exclamation': bool,
                'sentiment_score': Dict[str, float],
                'emotions': Dict[str, float],
                'topics': List[str],
                'keywords': List[str]
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
        
        # Obter analisador (lazy loading)
        analyzer = self._get_analyzer()
        
        # Executar análises em paralelo (futuro: usar asyncio.gather)
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
                logger.debug(
                    "🧠 [ANÁLISE] Executando análise semântica com SBERT",
                    meeting_id=chunk.meetingId
                )
                # Realizar análise semântica completa
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
            'embedding': embedding
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

