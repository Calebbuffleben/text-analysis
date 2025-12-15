"""
Analisador de texto usando modelo BERT pré-treinado em português.
Implementa análise de sentimento, extração de keywords, detecção de emoções
e análise semântica com SBERT (Sentence-BERT).
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple, Any
import structlog
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

logger = structlog.get_logger()

# ============================================================================
# EXEMPLOS DE REFERÊNCIA PARA CLASSIFICAÇÃO DE CATEGORIAS DE VENDAS
# ============================================================================
# 
# Estes exemplos são usados para classificação semântica de textos em categorias
# específicas de contexto de vendas/negócios usando SBERT (Sentence-BERT).
# 
# Como funciona:
# 1. Cada categoria possui múltiplos exemplos de texto representativos
# 2. Os embeddings desses exemplos são pré-calculados e armazenados em cache
# 3. Quando um novo texto precisa ser classificado:
#    a) Gera-se o embedding do texto usando SBERT
#    b) Calcula-se a similaridade semântica (cosseno) com cada exemplo de cada categoria
#    c) A categoria com maior similaridade média é selecionada
#    d) A confiança é calculada baseada na diferença entre a melhor e segunda melhor categoria
#
# Por que usar exemplos múltiplos?
# - Aumenta robustez: diferentes formas de expressar a mesma intenção
# - Melhora precisão: captura variações linguísticas e contextuais
# - Reduz falsos positivos: textos ambíguos tendem a ter similaridade baixa com todos os exemplos
#
# Estrutura:
# - Cada categoria é uma chave do dicionário
# - Cada valor é uma lista de strings (exemplos de texto em português)
# - Exemplos devem ser representativos e variados para melhor classificação
# ============================================================================

SALES_CATEGORY_EXAMPLES: Dict[str, List[str]] = {
    'price_interest': [
        # Cliente demonstra interesse explícito em saber o preço
        "Quanto custa isso?",
        "Qual é o preço?",
        "Quanto eu vou pagar por isso?",
        "Qual o valor desse produto?",
        "Preciso saber o preço",
        "Quanto sai isso?",
        "Qual o custo?",
        "Me fale sobre o preço",
        "Quanto fica?",
        "Preciso do valor",
        # Variações regionais e informais
        "Quanto é?",
        "Qual o preço disso?",
        "Quanto vou gastar?",
        "Me passa o valor",
        "Quanto custa em média?"
    ],
    
    'value_exploration': [
        # Cliente explora o valor e benefícios da solução
        "Como isso vai me ajudar?",
        "Qual o benefício disso para mim?",
        "Por que isso é melhor que outras opções?",
        "Como funciona essa solução?",
        "Me explique o valor que isso traz",
        "O que isso resolve para mim?",
        "Como isso melhora minha situação?",
        "Quais são as vantagens?",
        "Por que eu deveria escolher isso?",
        "Como isso se diferencia?",
        # Variações e formas indiretas
        "O que isso pode fazer por mim?",
        "Como isso agrega valor?",
        "Qual o diferencial dessa solução?",
        "Por que vale a pena investir nisso?",
        "Como isso se compara com outras alternativas?"
    ],
    
    'objection_soft': [
        # Objeções leves, dúvidas ou hesitações não definitivas
        "Não tenho certeza se preciso disso",
        "Preciso pensar melhor",
        "Talvez depois eu considere",
        "Não sei se é para mim",
        "Acho que não é o momento certo",
        "Vou precisar avaliar melhor",
        "Não estou completamente convencido",
        "Tenho algumas dúvidas",
        "Preciso conversar com outras pessoas",
        "Não tenho pressa para decidir",
        # Variações e formas mais sutis
        "Não tenho tanta certeza",
        "Preciso refletir sobre isso",
        "Vou considerar com calma",
        "Tenho minhas ressalvas",
        "Não estou totalmente seguro"
    ],
    
    'objection_hard': [
        # Objeções fortes e definitivas, rejeição clara
        "Não estou interessado",
        "Não preciso disso",
        "Muito caro para mim",
        "Não funciona para minha situação",
        "Não quero isso",
        "Não é o que eu procuro",
        "Não faz sentido para mim",
        "Não tenho interesse",
        "Não é prioridade",
        "Não vou comprar",
        # Variações mais diretas e definitivas
        "Não me interessa",
        "Não é para mim",
        "Não vou contratar",
        "Não preciso dessa solução",
        "Não é o que estou procurando"
    ],
    
    'decision_signal': [
        # Sinais claros de que o cliente está pronto para tomar decisão
        "Quando posso começar?",
        "Como faço para contratar?",
        "Quero isso",
        "Vamos fechar o negócio",
        "Estou pronto para avançar",
        "Como procedemos?",
        "Quero seguir em frente",
        "Vamos fazer isso",
        "Estou convencido",
        "Quero contratar",
        # Variações e formas de expressar decisão
        "Vamos em frente",
        "Estou decidido",
        "Quero começar logo",
        "Como fazemos para iniciar?",
        "Estou pronto para começar"
    ],
    
    'information_gathering': [
        # Cliente busca informações adicionais sobre a solução
        "Me explique mais sobre isso",
        "Como funciona exatamente?",
        "Quais são as opções disponíveis?",
        "Preciso de mais informações",
        "Conte-me mais detalhes",
        "Como isso se integra?",
        "Quais são os requisitos?",
        "Preciso entender melhor",
        "Me dê mais detalhes",
        "Explique melhor como funciona",
        # Variações e formas de buscar informação
        "Quero saber mais",
        "Me conte mais sobre",
        "Preciso entender como funciona",
        "Quais são os detalhes?",
        "Como é o processo?"
    ],
    
    'stalling': [
        # Cliente está protelando ou adiando a decisão
        "Deixa eu ver",
        "Vou pensar sobre isso",
        "Preciso consultar minha equipe",
        "Não tenho pressa para decidir",
        "Depois eu decido",
        "Vou avaliar melhor",
        "Preciso de mais tempo",
        "Não é urgente",
        "Vou considerar depois",
        "Ainda não sei",
        # Variações e formas de protelar
        "Deixa eu pensar",
        "Vou ver depois",
        "Não tenho pressa",
        "Vou analisar com calma",
        "Preciso de um tempo para pensar"
    ],
    
    'closing_readiness': [
        # Cliente demonstra prontidão para fechar o negócio
        "Estou pronto para fechar",
        "Vamos fazer isso acontecer",
        "Quero avançar com isso",
        "Estou convencido da solução",
        "Vamos contratar",
        "Estou decidido",
        "Quero seguir adiante",
        "Vamos fechar",
        "Estou pronto",
        "Quero começar",
        # Variações e formas mais explícitas
        "Vamos fechar o negócio",
        "Estou totalmente pronto",
        "Quero fechar agora",
        "Vamos contratar isso",
        "Estou completamente decidido"
    ]
}

# Mapeamento de progressão de categorias de vendas
# Usado para detectar transições e calcular tendências semânticas
# Valores positivos = progressão (avançando)
# Valores negativos = regressão (voltando)
# Zero = neutro/estagnado
CATEGORY_PROGRESSION: Dict[str, int] = {
    'information_gathering': 1,    # Estágio inicial: coletando informações
    'value_exploration': 2,        # Explorando valor e benefícios
    'price_interest': 3,           # Interesse em preço (avançando)
    'decision_signal': 4,         # Sinais de decisão
    'closing_readiness': 5,        # Pronto para fechar (estágio mais avançado)
    'stalling': 0,                 # Neutro: protelando
    'objection_soft': -1,          # Regressão leve: objeções suaves
    'objection_hard': -2           # Regressão forte: objeções duras
}

# Baixar recursos NLTK se necessário
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class BERTAnalyzer:
    """
    Analisador de texto usando BERT para português.
    
    Funcionalidades:
    - Análise de sentimento (positivo/negativo/neutro)
    - Extração de keywords
    - Detecção básica de emoções
    - Análise semântica com SBERT (embeddings e similaridade)
    - Classificação de categorias de vendas usando SBERT (price_interest, value_exploration,
      objection_soft, objection_hard, decision_signal, information_gathering, stalling,
      closing_readiness)
    
    A classificação de categorias de vendas utiliza exemplos de referência pré-definidos
    e compara semanticamente o texto de entrada com esses exemplos usando embeddings SBERT.
    Os embeddings dos exemplos são pré-calculados e armazenados em cache para otimização
    de performance.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        cache_dir: str = None,
        max_length: int = 512,
        sbert_model_name: Optional[str] = None
    ):
        """
        Inicializa analisador BERT.
        
        Args:
            model_name: Nome do modelo BERT no Hugging Face Hub
            device: Dispositivo ('cpu' ou 'cuda')
            cache_dir: Diretório para cache de modelos
            max_length: Tamanho máximo de tokens
            sbert_model_name: Nome do modelo SBERT (opcional, para análise semântica)
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.sbert_model_name = sbert_model_name
        
        # Modelos BERT para análise de sentimento
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._loaded = False
        
        # Modelo SBERT para análise semântica (lazy loading)
        # SBERT é uma arquitetura especializada em gerar embeddings semânticos
        # de sentenças completas, otimizada para tarefas de similaridade semântica
        self.sbert_model: Optional[SentenceTransformer] = None
        self._sbert_loaded = False
        
        # ========================================================================
        # CACHE DE EMBEDDINGS PARA CLASSIFICAÇÃO DE CATEGORIAS DE VENDAS
        # ========================================================================
        # 
        # Estrutura de dados para armazenar embeddings pré-calculados dos exemplos
        # de referência de cada categoria de vendas.
        #
        # Por que cachear embeddings?
        # ----------------------------
        # - Performance: Calcular embeddings é custoso (~50ms por texto)
        # - Eficiência: Os exemplos são fixos, não precisam ser recalculados
        # - Escalabilidade: Uma vez calculados, podem ser reutilizados infinitamente
        #
        # Estrutura de dados:
        # -------------------
        # _sales_category_examples_embeddings: Dict[str, List[np.ndarray]]
        #   - Chave: nome da categoria (ex: 'price_interest')
        #   - Valor: lista de arrays numpy, cada um representando o embedding
        #            de um exemplo de texto dessa categoria
        #   - Exemplo:
        #     {
        #       'price_interest': [array([0.1, 0.2, ...]), array([0.3, 0.4, ...]), ...],
        #       'value_exploration': [array([0.5, 0.6, ...]), ...],
        #       ...
        #     }
        #
        # _sales_examples_loaded: bool
        #   - Flag para indicar se os embeddings dos exemplos já foram calculados
        #   - Evita recalcular embeddings desnecessariamente (lazy loading)
        #   - Inicializado como False, torna-se True após primeiro carregamento
        #
        # Quando são carregados?
        # ----------------------
        # - Lazy loading: apenas quando necessário (primeira chamada de classificação)
        # - Requer que SBERT esteja carregado (_sbert_loaded = True)
        # - Calculado uma única vez e reutilizado para todas as classificações
        #
        # Benefícios:
        # -----------
        # - Reduz latência de classificação de ~50ms para ~5ms por texto
        # - Permite processar múltiplas classificações rapidamente
        # - Não impacta memória significativamente (8 categorias × ~10 exemplos × 384 dims ≈ 30KB)
        # ========================================================================
        self._sales_category_examples_embeddings: Optional[Dict[str, List[np.ndarray]]] = None
        self._sales_examples_loaded = False
        
        # Stopwords em português
        try:
            self.stopwords = set(stopwords.words('portuguese'))
        except Exception as e:
            logger.warn("Failed to load Portuguese stopwords, using fallback", error=str(e))
            self.stopwords = set([
                'o', 'a', 'de', 'para', 'com', 'em', 'um', 'uma', 'que', 'é',
                'do', 'da', 'no', 'na', 'os', 'as', 'dos', 'das', 'nos', 'nas'
            ])
        
        logger.info(
            "BERTAnalyzer initialized",
            model=model_name,
            device=device,
            cache_dir=cache_dir
        )
    
    def _load_model(self):
        """Carrega modelo BERT (lazy loading)"""
        if self._loaded:
            return
        
        logger.info("Loading BERT model", model=self.model_name)
        
        try:
            # Carregar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Carregar modelo
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Mover para dispositivo
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                self.device = "cuda"
                logger.info("Using CUDA device")
            else:
                self.device = "cpu"
                logger.info("Using CPU device")
            
            # Modo de avaliação (não treinamento)
            self.model.eval()
            
            # Criar pipeline para análise de sentimento
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=False
            )
            
            self._loaded = True
            logger.info("BERT model loaded successfully", device=self.device)
            
        except Exception as e:
            logger.error("Failed to load BERT model", error=str(e))
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analisa sentimento do texto usando BERT.
        
        Args:
            text: Texto a ser analisado
            
        Returns:
            Dict com scores de sentimento:
            {
                'positive': float,
                'negative': float,
                'neutral': float
            }
        """
        if not self._loaded:
            self._load_model()
        
        try:
            # Truncar texto se muito longo
            if len(text) > self.max_length * 4:  # Aproximação: 4 chars por token
                text = text[:self.max_length * 4]
                logger.debug("Text truncated", original_length=len(text))
            
            # Análise com pipeline
            result = self.pipeline(text)[0]
            
            # Extrair label e score
            label = result['label'].lower()
            score = result['score']
            
            # Normalizar para formato esperado
            sentiment = {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 0.0
            }
            
            # Mapear labels do modelo para nosso formato
            if 'pos' in label or 'positive' in label or 'positivo' in label:
                sentiment['positive'] = score
                sentiment['neutral'] = 1.0 - score
            elif 'neg' in label or 'negative' in label or 'negativo' in label:
                sentiment['negative'] = score
                sentiment['neutral'] = 1.0 - score
            else:
                # Label neutro ou desconhecido
                sentiment['neutral'] = score
                sentiment['positive'] = (1.0 - score) / 2
                sentiment['negative'] = (1.0 - score) / 2
            
            logger.debug(
                "Sentiment analysis completed",
                sentiment=sentiment,
                label=label
            )
            
            return sentiment
            
        except Exception as e:
            logger.error("Sentiment analysis failed", error=str(e), text_preview=text[:50])
            # Retornar sentimento neutro em caso de erro
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extrai palavras-chave do texto usando NLTK.
        
        Args:
            text: Texto a ser processado
            top_n: Número de keywords a retornar
            
        Returns:
            Lista de palavras-chave ordenadas por frequência
        """
        try:
            # Normalizar texto
            text_lower = text.lower()
            
            # Tokenizar
            words = word_tokenize(text_lower, language='portuguese')
            
            # Filtrar: remover stopwords, pontuação e palavras muito curtas
            filtered_words = [
                word for word in words
                if word.isalnum()
                and word not in self.stopwords
                and len(word) > 2
            ]
            
            # Contar frequência
            word_freq = Counter(filtered_words)
            
            # Retornar top N
            keywords = [word for word, _ in word_freq.most_common(top_n)]
            
            logger.debug(
                "Keywords extracted",
                count=len(keywords),
                keywords=keywords[:5]
            )
            
            return keywords
            
        except Exception as e:
            logger.error("Keyword extraction failed", error=str(e))
            return []
    
    def detect_emotions(self, text: str) -> Dict[str, float]:
        """
        Detecta emoções básicas no texto (versão simplificada baseada em keywords).
        
        Args:
            text: Texto a ser analisado
            
        Returns:
            Dict com scores de emoções:
            {
                'joy': float,
                'sadness': float,
                'anger': float,
                'fear': float,
                'surprise': float
            }
        """
        emotions = {
            'joy': 0.0,
            'sadness': 0.0,
            'anger': 0.0,
            'fear': 0.0,
            'surprise': 0.0
        }
        
        # Palavras-chave para cada emoção (português)
        emotion_keywords = {
            'joy': [
                'feliz', 'alegre', 'content', 'satisfeito', 'animado',
                'entusiasmado', 'empolgado', 'radiante', 'eufórico'
            ],
            'sadness': [
                'triste', 'deprimido', 'desanimado', 'melancólico',
                'chateado', 'desapontado', 'abatido', 'desolado'
            ],
            'anger': [
                'raiva', 'irritado', 'furioso', 'bravo', 'nervoso',
                'irritado', 'revoltado', 'indignado', 'exasperado'
            ],
            'fear': [
                'medo', 'assustado', 'preocupado', 'ansioso', 'temor',
                'pânico', 'apreensivo', 'receoso', 'amedrontado'
            ],
            'surprise': [
                'surpreso', 'impressionado', 'chocado', 'admirado',
                'espantado', 'atônito', 'maravilhado', 'deslumbrado'
            ]
        }
        
        text_lower = text.lower()
        
        # Contar ocorrências de keywords para cada emoção
        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if count > 0:
                # Normalizar score (0 a 1)
                emotions[emotion] = min(count / len(keywords), 1.0)
        
        logger.debug("Emotions detected", emotions=emotions)
        
        return emotions
    
    def _load_sbert_model(self):
        """
        Carrega modelo SBERT para análise semântica (lazy loading).
        
        SBERT (Sentence-BERT) é uma variação do BERT otimizada para:
        - Gerar embeddings semânticos de sentenças completas
        - Calcular similaridade semântica entre textos
        - Operações de busca semântica e clustering
        
        O modelo escolhido (paraphrase-multilingual-MiniLM-L12-v2) é:
        - Multilíngue (suporta português)
        - Leve e rápido (MiniLM)
        - Otimizado para similaridade semântica
        
        O embedding gerado é um vetor denso de dimensão fixa (384 no caso deste modelo)
        que representa o significado semântico do texto. Textos com significados similares
        terão embeddings próximos no espaço vetorial.
        """
        if self._sbert_loaded or not self.sbert_model_name:
            return
        
        logger.info("Loading SBERT model for semantic analysis", model=self.sbert_model_name)
        
        try:
            # SentenceTransformer gerencia automaticamente:
            # - Carregamento do tokenizer
            # - Carregamento do modelo base
            # - Pooling de embeddings (mean pooling por padrão)
            # - Normalização de embeddings
            self.sbert_model = SentenceTransformer(
                self.sbert_model_name,
                cache_folder=self.cache_dir,
                device=self.device
            )
            
            # Mover para dispositivo correto se necessário
            if self.device == "cuda" and torch.cuda.is_available():
                self.sbert_model = self.sbert_model.to("cuda")
                logger.info("SBERT model loaded on CUDA")
            else:
                logger.info("SBERT model loaded on CPU")
            
            self._sbert_loaded = True
            
            # Log da dimensão do embedding para referência
            # Isso ajuda a entender o tamanho dos vetores gerados
            sample_embedding = self.sbert_model.encode("test", convert_to_numpy=True)
            embedding_dim = len(sample_embedding)
            
            logger.info(
                "SBERT model loaded successfully",
                model=self.sbert_model_name,
                device=self.device,
                embedding_dimension=embedding_dim
            )
            
        except Exception as e:
            logger.error("Failed to load SBERT model", error=str(e), model=self.sbert_model_name)
            raise
    
    def generate_semantic_embedding(self, text: str) -> np.ndarray:
        """
        Gera embedding semântico para um texto usando SBERT.
        
        O que é um embedding semântico?
        ================================
        Um embedding é uma representação numérica densa de um texto em um espaço
        vetorial de alta dimensão. Textos com significados similares terão embeddings
        próximos (medidos por distância cosseno ou euclidiana).
        
        Como funciona o SBERT?
        =======================
        1. Tokenização: O texto é dividido em tokens (palavras/subpalavras)
        2. Encoding: Cada token recebe um embedding do modelo BERT
        3. Pooling: Os embeddings dos tokens são agregados (mean pooling) para
           gerar um único vetor representando a sentença completa
        4. Normalização: O vetor é normalizado (L2 norm) para facilitar comparações
        
        Exemplo de uso:
        ===============
        embedding1 = analyzer.generate_semantic_embedding("Estou feliz hoje")
        embedding2 = analyzer.generate_semantic_embedding("Me sinto alegre")
        # embedding1 e embedding2 serão similares (alta similaridade cosseno)
        
        Args:
            text: Texto para gerar embedding
            
        Returns:
            numpy.ndarray: Vetor de embedding normalizado (dimensão fixa, tipicamente 384)
            
        Raises:
            RuntimeError: Se o modelo SBERT não estiver disponível
        """
        if not self.sbert_model_name:
            raise RuntimeError("SBERT model not configured. Set sbert_model_name in config.")
        
        if not self._sbert_loaded:
            self._load_sbert_model()
        
        try:
            # Truncar texto se muito longo
            # Modelos SBERT têm limite de tokens (geralmente 512)
            if len(text) > self.max_length * 4:
                text = text[:self.max_length * 4]
                logger.debug("Text truncated for semantic embedding", original_length=len(text))
            
            # Gerar embedding
            # O método encode() retorna um array numpy com o embedding normalizado
            # convert_to_numpy=True garante que retornamos um array numpy ao invés de tensor PyTorch
            embedding = self.sbert_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Normalização L2 para facilitar comparações
                show_progress_bar=False
            )
            
            logger.debug(
                "Semantic embedding generated",
                text_preview=text[:50],
                embedding_dim=len(embedding),
                embedding_norm=np.linalg.norm(embedding)  # Deve ser ~1.0 devido à normalização
            )
            
            return embedding
            
        except Exception as e:
            logger.error(
                "Failed to generate semantic embedding",
                error=str(e),
                text_preview=text[:50]
            )
            raise
    
    def calculate_semantic_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Calcula similaridade semântica entre dois textos usando SBERT.
        
        Como funciona a similaridade semântica?
        =======================================
        A similaridade semântica mede o quão próximos são os significados de dois textos,
        independentemente das palavras exatas usadas. Por exemplo:
        
        - "Estou feliz" e "Me sinto alegre" → Alta similaridade (~0.85)
        - "Estou feliz" e "Estou triste" → Baixa similaridade (~0.30)
        - "Gato" e "Cachorro" → Similaridade média (~0.60, ambos são animais)
        
        Método de cálculo:
        ==================
        Usamos Similaridade de Cosseno (Cosine Similarity):
        
        similarity = cos(θ) = (A · B) / (||A|| × ||B||)
        
        Onde:
        - A e B são os embeddings dos textos
        - · é o produto escalar
        - ||A|| e ||B|| são as normas dos vetores
        
        Como os embeddings são normalizados (L2 norm = 1), a fórmula simplifica para:
        
        similarity = A · B  (produto escalar)
        
        Resultado:
        =========
        - 1.0: Textos idênticos ou semanticamente equivalentes
        - 0.7-0.9: Textos muito similares (paráfrases, sinônimos)
        - 0.5-0.7: Textos relacionados mas diferentes
        - 0.0-0.5: Textos não relacionados ou opostos
        
        Args:
            text1: Primeiro texto para comparação
            text2: Segundo texto para comparação
            
        Returns:
            float: Score de similaridade entre 0.0 e 1.0
                  (1.0 = idêntico, 0.0 = completamente diferente)
                  
        Raises:
            RuntimeError: Se o modelo SBERT não estiver disponível
        """
        if not self.sbert_model_name:
            raise RuntimeError("SBERT model not configured. Set sbert_model_name in config.")
        
        if not self._sbert_loaded:
            self._load_sbert_model()
        
        try:
            # Gerar embeddings para ambos os textos
            embedding1 = self.generate_semantic_embedding(text1)
            embedding2 = self.generate_semantic_embedding(text2)
            
            # Calcular similaridade de cosseno
            # Como os embeddings são normalizados, o produto escalar é igual à similaridade de cosseno
            similarity = float(np.dot(embedding1, embedding2))
            
            # Garantir que o resultado está no range [0, 1]
            # (embora teoricamente deveria estar, devido à normalização)
            similarity = max(0.0, min(1.0, similarity))
            
            logger.debug(
                "Semantic similarity calculated",
                text1_preview=text1[:30],
                text2_preview=text2[:30],
                similarity=round(similarity, 4)
            )
            
            return similarity
            
        except Exception as e:
            logger.error(
                "Failed to calculate semantic similarity",
                error=str(e),
                text1_preview=text1[:30],
                text2_preview=text2[:30]
            )
            # Retornar 0.0 em caso de erro (textos não relacionados)
            return 0.0
    
    def analyze_semantics(
        self,
        text: str,
        reference_texts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Realiza análise semântica completa do texto.
        
        Esta função combina várias técnicas de análise semântica:
        1. Geração de embedding semântico
        2. Cálculo de similaridade com textos de referência (se fornecidos)
        3. Extração de características semânticas
        
        Casos de uso:
        =============
        - Detectar repetição de ideias (similaridade alta com textos anteriores)
        - Identificar mudanças de tópico (similaridade baixa)
        - Agrupar transcrições por tópico semântico
        - Detectar paráfrases ou reformulações
        
        Args:
            text: Texto a ser analisado
            reference_texts: Lista opcional de textos de referência para calcular similaridade
            
        Returns:
            Dict com resultados da análise semântica:
            {
                'embedding_dimension': int,  # Dimensão do vetor de embedding
                'embedding_norm': float,     # Norma L2 do embedding (deve ser ~1.0)
                'similarities': List[float], # Similaridades com textos de referência (se fornecidos)
                'avg_similarity': float,     # Média das similaridades (se houver referências)
                'max_similarity': float,      # Máxima similaridade (se houver referências)
                'min_similarity': float       # Mínima similaridade (se houver referências)
            }
            
        Raises:
            RuntimeError: Se o modelo SBERT não estiver disponível
        """
        if not self.sbert_model_name:
            raise RuntimeError("SBERT model not configured. Set sbert_model_name in config.")
        
        if not self._sbert_loaded:
            self._load_sbert_model()
        
        try:
            # Gerar embedding semântico
            embedding = self.generate_semantic_embedding(text)
            
            # Inicializar resultado
            result = {
                'embedding_dimension': len(embedding),
                'embedding_norm': float(np.linalg.norm(embedding)),
            }
            
            # Calcular similaridades com textos de referência (se fornecidos)
            if reference_texts and len(reference_texts) > 0:
                similarities = []
                
                for ref_text in reference_texts:
                    similarity = self.calculate_semantic_similarity(text, ref_text)
                    similarities.append(similarity)
                
                result['similarities'] = similarities
                result['avg_similarity'] = float(np.mean(similarities))
                result['max_similarity'] = float(np.max(similarities))
                result['min_similarity'] = float(np.min(similarities))
                
                logger.debug(
                    "Semantic analysis completed with references",
                    text_preview=text[:50],
                    num_references=len(reference_texts),
                    avg_similarity=round(result['avg_similarity'], 4)
                )
            else:
                # Sem textos de referência, apenas retornar informações do embedding
                result['similarities'] = []
                result['avg_similarity'] = None
                result['max_similarity'] = None
                result['min_similarity'] = None
                
                logger.debug(
                    "Semantic analysis completed",
                    text_preview=text[:50],
                    embedding_dim=result['embedding_dimension']
                )
            
            return result
            
        except Exception as e:
            logger.error(
                "Semantic analysis failed",
                error=str(e),
                text_preview=text[:50]
            )
            # Retornar resultado vazio em caso de erro
            return {
                'embedding_dimension': 0,
                'embedding_norm': 0.0,
                'similarities': [],
                'avg_similarity': None,
                'max_similarity': None,
                'min_similarity': None
            }
    
    def _load_sales_category_examples_embeddings(self):
        """
        Pré-calcula e armazena em cache os embeddings dos exemplos de referência
        para cada categoria de vendas.
        
        Este método implementa lazy loading: os embeddings são calculados apenas
        uma vez, na primeira chamada, e depois reutilizados para todas as classificações
        subsequentes. Isso otimiza significativamente a performance.
        
        Como funciona:
        ==============
        1. Verifica se os embeddings já foram carregados (_sales_examples_loaded)
        2. Verifica se o modelo SBERT está disponível e carregado
        3. Para cada categoria em SALES_CATEGORY_EXAMPLES:
           a) Para cada exemplo de texto dessa categoria:
              - Gera o embedding usando SBERT
              - Armazena o embedding na lista da categoria
        4. Armazena todos os embeddings em _sales_category_examples_embeddings
        5. Marca _sales_examples_loaded como True
        
        Performance:
        ============
        - Primeira chamada: ~400-500ms (calcula ~80 embeddings)
        - Chamadas subsequentes: ~0ms (usa cache)
        - Memória: ~30KB (8 categorias × 10 exemplos × 384 dims × 4 bytes)
        
        Otimizações:
        ============
        - Usa encode() em batch quando possível (futuro)
        - Embeddings são normalizados (L2 norm = 1) para comparações eficientes
        - Cache persiste durante toda a vida útil da instância
        
        Raises:
        =======
        RuntimeError: Se o modelo SBERT não estiver configurado ou disponível
        
        Exemplo de uso interno:
        ======================
        Este método é chamado automaticamente por classify_sales_category() quando
        necessário. Não precisa ser chamado manualmente.
        """
        # Verificar se já foi carregado (evitar recálculo desnecessário)
        if self._sales_examples_loaded:
            logger.debug("Sales category examples embeddings already loaded, skipping")
            return
        
        # Verificar se SBERT está configurado
        if not self.sbert_model_name:
            raise RuntimeError(
                "SBERT model not configured. Cannot load sales category examples embeddings. "
                "Set sbert_model_name in config."
            )
        
        # Garantir que o modelo SBERT está carregado
        if not self._sbert_loaded:
            logger.info("SBERT model not loaded yet, loading now for sales category classification")
            self._load_sbert_model()
        
        logger.info(
            "Loading sales category examples embeddings",
            num_categories=len(SALES_CATEGORY_EXAMPLES),
            total_examples=sum(len(examples) for examples in SALES_CATEGORY_EXAMPLES.values())
        )
        
        try:
            # Inicializar dicionário para armazenar embeddings por categoria
            # Estrutura: {categoria: [embedding1, embedding2, ...]}
            embeddings_cache: Dict[str, List[np.ndarray]] = {}
            
            # Processar cada categoria
            for category_name, example_texts in SALES_CATEGORY_EXAMPLES.items():
                logger.debug(
                    "Processing category",
                    category=category_name,
                    num_examples=len(example_texts)
                )
                
                # Lista para armazenar embeddings desta categoria
                category_embeddings: List[np.ndarray] = []
                
                # Gerar embedding para cada exemplo de texto desta categoria
                for example_text in example_texts:
                    # Gerar embedding usando SBERT
                    # O método generate_semantic_embedding() já faz:
                    # - Truncamento se necessário
                    # - Normalização L2
                    # - Conversão para numpy array
                    embedding = self.generate_semantic_embedding(example_text)
                    
                    # Armazenar embedding na lista da categoria
                    category_embeddings.append(embedding)
                
                # Armazenar lista de embeddings desta categoria no cache
                embeddings_cache[category_name] = category_embeddings
                
                logger.debug(
                    "Category embeddings loaded",
                    category=category_name,
                    num_embeddings=len(category_embeddings),
                    embedding_dim=len(category_embeddings[0]) if category_embeddings else 0
                )
            
            # Armazenar cache completo no atributo da instância
            self._sales_category_examples_embeddings = embeddings_cache
            
            # Marcar como carregado (evita recálculo)
            self._sales_examples_loaded = True
            
            # Log de sucesso com estatísticas
            total_embeddings = sum(len(embeddings) for embeddings in embeddings_cache.values())
            logger.info(
                "Sales category examples embeddings loaded successfully",
                num_categories=len(embeddings_cache),
                total_embeddings=total_embeddings,
                embedding_dimension=len(embeddings_cache[list(embeddings_cache.keys())[0]][0]) if embeddings_cache else 0
            )
            
        except Exception as e:
            logger.error(
                "Failed to load sales category examples embeddings",
                error=str(e),
                error_type=type(e).__name__
            )
            # Limpar estado parcial em caso de erro
            self._sales_category_examples_embeddings = None
            self._sales_examples_loaded = False
            raise RuntimeError(
                f"Failed to load sales category examples embeddings: {str(e)}"
            ) from e
    
    def _generate_semantic_flags(
        self,
        category: Optional[str],
        confidence: float,
        intensity: float,
        ambiguity: float
    ) -> Dict[str, bool]:
        """
        Gera flags semânticas booleanas baseadas em análise completa.
        
        Flags são sinais booleanos que facilitam decisões no backend sem precisar
        interpretar múltiplos scores e thresholds. Cada flag indica uma condição
        específica que pode ser usada diretamente em heurísticas de negócio.
        
        Args:
        =====
        category: Optional[str]
            Categoria detectada (None se nenhuma)
        confidence: float
            Confiança da classificação (0.0 a 1.0)
        intensity: float
            Intensidade do sinal (score absoluto da melhor categoria)
        ambiguity: float
            Ambiguidade semântica (0.0 = claro, 1.0 = muito ambíguo)
        
        Returns:
        ========
        Dict[str, bool]
            Dicionário com flags booleanas:
            - price_window_open: True se há janela de oportunidade para falar sobre preço
            - decision_signal_strong: True se há sinal forte de que cliente está pronto para decidir
            - ready_to_close: True se cliente demonstra prontidão para fechar o negócio
        
        Exemplos:
        =========
        >>> flags = analyzer._generate_semantic_flags('price_interest', 0.85, 0.9, 0.2)
        >>> flags['price_window_open']  # True (alta confiança, alta intensidade, baixa ambiguidade)
        
        >>> flags = analyzer._generate_semantic_flags('closing_readiness', 0.9, 0.95, 0.15)
        >>> flags['ready_to_close']  # True (múltiplos critérios atendidos)
        """
        flags: Dict[str, bool] = {}
        
        if not category:
            return flags
        
        # Flag: Janela de oportunidade para preço
        # Indica que é o momento ideal para apresentar o preço
        # Requisitos: categoria price_interest + alta confiança + alta intensidade + baixa ambiguidade
        flags['price_window_open'] = (
            category == 'price_interest' and
            confidence > 0.7 and
            intensity > 0.8 and
            ambiguity < 0.3
        )
        
        # Flag: Sinal forte de decisão
        # Indica que cliente está demonstrando sinais claros de prontidão para decidir
        # Requisitos: categoria decision_signal ou closing_readiness + alta confiança + alta intensidade + baixa ambiguidade
        flags['decision_signal_strong'] = (
            category in ['decision_signal', 'closing_readiness'] and
            confidence > 0.8 and
            intensity > 0.85 and
            ambiguity < 0.25
        )
        
        # Flag: Pronto para fechar
        # Indica que cliente demonstra prontidão explícita para fechar o negócio
        # Requisitos: categoria closing_readiness + muito alta confiança + muito alta intensidade + muito baixa ambiguidade
        flags['ready_to_close'] = (
            category == 'closing_readiness' and
            confidence > 0.85 and
            intensity > 0.9 and
            ambiguity < 0.2
        )
        
        return flags
    
    def _calculate_ambiguity(self, scores: Dict[str, float]) -> float:
        """
        Calcula ambiguidade semântica baseada na distribuição dos scores.
        
        A ambiguidade é calculada usando entropia normalizada dos scores.
        Alta ambiguidade indica que múltiplas categorias têm scores similares,
        sugerindo que o texto pode ser interpretado de várias formas.
        
        Args:
        =====
        scores: Dict[str, float]
            Dicionário com scores de todas as categorias
        
        Returns:
        ========
        float: Score de ambiguidade de 0.0 (claro, uma categoria dominante) 
               a 1.0 (muito ambíguo, scores muito próximos entre categorias)
        
        Exemplos:
        =========
        >>> scores_clear = {'price_interest': 0.9, 'value_exploration': 0.1}
        >>> ambiguity = analyzer._calculate_ambiguity(scores_clear)
        >>> # ambiguity será baixa (< 0.3)
        
        >>> scores_ambiguous = {'price_interest': 0.5, 'value_exploration': 0.48}
        >>> ambiguity = analyzer._calculate_ambiguity(scores_ambiguous)
        >>> # ambiguity será alta (> 0.7)
        """
        if not scores:
            return 1.0  # Sem scores = máxima ambiguidade
        
        sorted_scores = sorted(scores.values(), reverse=True)
        
        if len(sorted_scores) < 2:
            return 0.0  # Apenas uma categoria = sem ambiguidade
        
        # Calcular entropia normalizada dos scores
        # Entropia alta = scores distribuídos uniformemente = alta ambiguidade
        # Entropia baixa = uma categoria dominante = baixa ambiguidade
        import numpy as np
        scores_array = np.array(sorted_scores)
        
        # Normalizar scores para probabilidades (soma = 1)
        scores_sum = scores_array.sum()
        if scores_sum == 0:
            return 1.0  # Todos zeros = máxima ambiguidade
        
        scores_normalized = scores_array / scores_sum
        
        # Calcular entropia de Shannon
        # H = -Σ(p_i * log(p_i))
        # Adicionar epsilon pequeno para evitar log(0)
        epsilon = 1e-10
        entropy = -np.sum(scores_normalized * np.log(scores_normalized + epsilon))
        
        # Normalizar pela entropia máxima possível (log(n))
        max_entropy = np.log(len(scores))
        
        if max_entropy == 0:
            return 0.0
        
        # Normalizar para [0, 1]
        normalized_entropy = entropy / max_entropy
        
        return float(normalized_entropy)
    
    def classify_sales_category(
        self,
        text: str,
        min_confidence: float = 0.3
    ) -> Tuple[Optional[str], float, Dict[str, float], float, float, Dict[str, bool]]:
        """
        Classifica um texto em uma das categorias de vendas usando análise semântica com SBERT.
        
        Este método utiliza exemplos de referência pré-definidos para cada categoria e compara
        semanticamente o texto de entrada com esses exemplos usando embeddings SBERT.
        
        Algoritmo de classificação:
        ===========================
        1. Gera embedding semântico do texto de entrada usando SBERT
        2. Para cada categoria:
           a) Calcula similaridade semântica (cosseno) entre o embedding do texto e cada
              embedding dos exemplos dessa categoria
           b) Calcula a média das similaridades (representa quão similar o texto é à categoria)
        3. Seleciona a categoria com maior similaridade média
        4. Calcula confiança baseada na diferença entre a melhor e segunda melhor categoria
        5. Retorna categoria, confiança e scores de todas as categorias
        
        Cálculo de confiança:
        =====================
        A confiança é calculada usando a diferença entre a melhor categoria e a segunda melhor:
        
        confidence = (best_score - second_best_score) / best_score
        
        Exemplos:
        - Se best=0.8 e second=0.3 → confidence = (0.8-0.3)/0.8 = 0.625 (alta confiança)
        - Se best=0.6 e second=0.55 → confidence = (0.6-0.55)/0.6 = 0.083 (baixa confiança)
        
        A confiança também considera o score absoluto da melhor categoria:
        - Se best_score < min_confidence, retorna None (categoria não confiável)
        
        Args:
        =====
        text: str
            Texto a ser classificado (ex: "Quanto custa isso?")
        
        min_confidence: float, opcional (padrão: 0.3)
            Score mínimo necessário para aceitar uma classificação.
            Se a melhor categoria tiver score < min_confidence, retorna None.
            Valores típicos:
            - 0.3: Permissivo (aceita classificações mais fracas)
            - 0.5: Moderado (balanceado)
            - 0.7: Restritivo (apenas classificações muito claras)
        
        Returns:
        ========
        Tuple[Optional[str], float, Dict[str, float], float, float, Dict[str, bool]]
            Tupla contendo:
            - categoria: str ou None
                * Nome da categoria detectada (ex: 'price_interest')
                * None se nenhuma categoria atingir min_confidence
            - confiança: float
                * Score de confiança da classificação (0.0 a 1.0)
                * Baseado na diferença entre melhor e segunda melhor categoria
                * 0.0 se categoria for None
            - scores: Dict[str, float]
                * Dicionário com scores de todas as categorias
                * Formato: {'price_interest': 0.85, 'value_exploration': 0.12, ...}
                * Útil para debugging e análise detalhada
            - ambiguidade: float
                * Score de ambiguidade semântica (0.0 a 1.0)
                * 0.0 = claro (uma categoria dominante)
                * 1.0 = muito ambíguo (scores muito próximos entre categorias)
                * Calculado usando entropia normalizada dos scores
            - intensidade: float
                * Score absoluto da melhor categoria (0.0 a 1.0)
                * Diferente de confiança: representa quão forte é o match semântico
                * Útil para diferenciar entre match fraco mas claro vs match forte
            - flags: Dict[str, bool]
                * Dicionário com flags semânticas booleanas
                * Formato: {'price_window_open': True, 'decision_signal_strong': False, ...}
                * Facilita heurísticas no backend sem precisar interpretar múltiplos scores
        
        Raises:
        =======
        RuntimeError: Se o modelo SBERT não estiver configurado ou disponível
        
        Exemplo de uso:
        ===============
        >>> analyzer = BERTAnalyzer(sbert_model_name='sentence-transformers/...')
        >>> categoria, confianca, scores, ambiguidade, intensidade, flags = analyzer.classify_sales_category("Quanto custa isso?")
        >>> print(categoria)     # 'price_interest'
        >>> print(confianca)      # 0.85
        >>> print(ambiguidade)    # 0.15 (baixa ambiguidade)
        >>> print(intensidade)    # 0.92 (alta intensidade)
        >>> print(flags)         # {'price_window_open': True, 'decision_signal_strong': False, ...}
        >>> print(scores)         # {'price_interest': 0.92, 'value_exploration': 0.15, ...}
        
        Performance:
        ============
        - Primeira chamada: ~50ms (carrega embeddings dos exemplos + classifica)
        - Chamadas subsequentes: ~5ms (usa cache de embeddings)
        - Otimizado para processar múltiplas classificações rapidamente
        
        Notas:
        ======
        - Os embeddings dos exemplos são carregados automaticamente na primeira chamada
        - A classificação é baseada em similaridade semântica, não em palavras-chave
        - Textos ambíguos tendem a ter scores baixos em todas as categorias
        - A confiança ajuda a identificar quando a classificação é incerta
        """
        # Verificar se SBERT está configurado
        if not self.sbert_model_name:
            logger.warn(
                "SBERT model not configured, cannot classify sales category",
                text_preview=text[:50]
            )
            return None, 0.0, {}, 1.0, 0.0, {}  # Máxima ambiguidade, intensidade zero, flags vazias
        
        # Garantir que os embeddings dos exemplos estão carregados
        if not self._sales_examples_loaded:
            logger.debug("Loading sales category examples embeddings for first classification")
            self._load_sales_category_examples_embeddings()
        
        # Verificar se os embeddings foram carregados com sucesso
        if not self._sales_category_examples_embeddings:
            logger.error(
                "Sales category examples embeddings not available",
                text_preview=text[:50]
            )
            return None, 0.0, {}, 1.0, 0.0, {}  # Máxima ambiguidade, intensidade zero, flags vazias
        
        try:
            # Gerar embedding semântico do texto de entrada
            # Este embedding será comparado com os embeddings dos exemplos
            text_embedding = self.generate_semantic_embedding(text)
            
            # Dicionário para armazenar scores de similaridade média por categoria
            # Formato: {categoria: score_médio}
            category_scores: Dict[str, float] = {}
            
            # Calcular similaridade média para cada categoria
            for category_name, example_embeddings in self._sales_category_examples_embeddings.items():
                # Lista para armazenar similaridades com todos os exemplos desta categoria
                similarities: List[float] = []
                
                # Calcular similaridade com cada exemplo desta categoria
                for example_embedding in example_embeddings:
                    # Similaridade de cosseno entre embeddings normalizados
                    # Como ambos são normalizados (L2 norm = 1), o produto escalar é a similaridade
                    similarity = float(np.dot(text_embedding, example_embedding))
                    
                    # Garantir que está no range [0, 1] (teoricamente já está, mas por segurança)
                    similarity = max(0.0, min(1.0, similarity))
                    
                    similarities.append(similarity)
                
                # Calcular média das similaridades
                # A média representa quão similar o texto é à categoria como um todo
                # Usar média ao invés de máximo porque:
                # - É mais robusto a outliers
                # - Captura melhor a semelhança geral com a categoria
                # - Reduz impacto de exemplos muito específicos
                avg_similarity = float(np.mean(similarities))
                
                # Armazenar score desta categoria
                category_scores[category_name] = avg_similarity
                
                logger.debug(
                    "Category similarity calculated",
                    category=category_name,
                    avg_similarity=round(avg_similarity, 4),
                    min_similarity=round(float(np.min(similarities)), 4),
                    max_similarity=round(float(np.max(similarities)), 4),
                    num_examples=len(similarities)
                )
            
            # Encontrar categoria com maior score (melhor match)
            if not category_scores:
                logger.warn("No category scores calculated", text_preview=text[:50])
                return None, 0.0, {}, 1.0, 0.0, {}  # Máxima ambiguidade, intensidade zero, flags vazias
            
            # Ordenar categorias por score (maior primeiro)
            sorted_categories = sorted(
                category_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            best_category, best_score = sorted_categories[0]
            
            # Calcular ambiguidade semântica
            # Quanto maior a ambiguidade, mais incerto é o texto
            ambiguity = self._calculate_ambiguity(category_scores)
            
            # Calcular intensidade (score absoluto da melhor categoria)
            # Diferente de confiança: representa quão forte é o match semântico
            intensity = best_score
            
            # Calcular confiança baseada na diferença entre melhor e segunda melhor categoria
            # Se houver apenas uma categoria, usar o próprio score como confiança
            if len(sorted_categories) > 1:
                second_best_score = sorted_categories[1][1]
                
                # Confiança = diferença relativa entre melhor e segunda melhor
                # Fórmula: (best - second) / best
                # Isso normaliza a diferença pelo score absoluto
                if best_score > 0:
                    confidence = (best_score - second_best_score) / best_score
                else:
                    confidence = 0.0
                
                # Adicionar um fator baseado no score absoluto da melhor categoria
                # Isso ajuda a aumentar confiança quando o score absoluto é alto
                # Fórmula ajustada: confidence = base_confidence * (1 + best_score) / 2
                # Isso dá mais peso quando best_score é alto
                confidence = confidence * (1.0 + best_score) / 2.0
                
                logger.debug(
                    "Confidence calculated",
                    best_category=best_category,
                    best_score=round(best_score, 4),
                    second_best_score=round(second_best_score, 4),
                    confidence=round(confidence, 4)
                )
            else:
                # Apenas uma categoria disponível (caso raro)
                confidence = best_score
            
            # Gerar flags semânticas baseadas na análise completa
            # (após calcular confiança, intensidade e ambiguidade)
            flags = self._generate_semantic_flags(best_category, confidence, intensity, ambiguity)
            
            # Verificar se o score da melhor categoria atinge o mínimo necessário
            if best_score < min_confidence:
                logger.debug(
                    "Best category score below minimum confidence threshold",
                    best_category=best_category,
                    best_score=round(best_score, 4),
                    min_confidence=min_confidence,
                    ambiguity=round(ambiguity, 4),
                    intensity=round(intensity, 4),
                    flags=flags,
                    text_preview=text[:50]
                )
                return None, 0.0, category_scores, ambiguity, intensity, flags
            
            # Garantir que confiança está no range [0, 1]
            confidence = max(0.0, min(1.0, confidence))
            
            logger.info(
                "Sales category classified",
                text_preview=text[:50],
                category=best_category,
                confidence=round(confidence, 4),
                best_score=round(best_score, 4),
                ambiguity=round(ambiguity, 4),
                intensity=round(intensity, 4),
                flags=flags,
                top_3_categories=[
                    (cat, round(score, 4))
                    for cat, score in sorted_categories[:3]
                ]
            )
            
            return best_category, confidence, category_scores, ambiguity, intensity, flags
            
        except Exception as e:
            logger.error(
                "Failed to classify sales category",
                error=str(e),
                error_type=type(e).__name__,
                text_preview=text[:50]
            )
            # Retornar valores padrão em caso de erro
            return None, 0.0, {}, 1.0, 0.0, {}  # Máxima ambiguidade, intensidade zero, flags vazias
    
    def classify_sales_category_multi(
        self,
        text: str,
        min_confidence: float = 0.3,
        max_categories: int = 2,
        similarity_threshold: float = 0.7
    ) -> Tuple[List[Tuple[str, float]], float, Dict[str, float]]:
        """
        Classifica texto em múltiplas categorias quando scores são próximos.
        
        Este método é útil quando um texto pode ser interpretado de múltiplas formas
        semanticamente válidas. Por exemplo, "Quanto custa e como funciona?" pode ser
        tanto 'price_interest' quanto 'information_gathering'.
        
        Algoritmo:
        ==========
        1. Usa classify_sales_category() para obter scores de todas as categorias
        2. Ordena categorias por score (maior primeiro)
        3. Se a segunda melhor categoria tem score >= similarity_threshold × melhor score
           E ambas acima de min_confidence, retorna múltiplas categorias
        4. Caso contrário, retorna apenas a melhor categoria
        
        Args:
        =====
        text: str
            Texto a ser classificado
        
        min_confidence: float, opcional (padrão: 0.3)
            Score mínimo necessário para aceitar uma categoria
        
        max_categories: int, opcional (padrão: 2)
            Máximo de categorias a retornar
        
        similarity_threshold: float, opcional (padrão: 0.7)
            Se segunda melhor categoria tem score >= threshold × melhor score,
            incluir ambas. Valores típicos:
            - 0.7: Permissivo (inclui categorias com scores próximos)
            - 0.8: Moderado (apenas categorias muito similares)
            - 0.9: Restritivo (apenas categorias quase idênticas)
        
        Returns:
        ========
        Tuple[List[Tuple[str, float]], float, Dict[str, float]]
            Tupla contendo:
            - categories: List[Tuple[str, float]]
                * Lista de (categoria, score) ordenada por score (maior primeiro)
                * Exemplo: [('price_interest', 0.75), ('information_gathering', 0.68)]
            - confidence: float
                * Confiança geral da classificação (0.0 a 1.0)
                * Baseado na diferença entre melhor e segunda melhor categoria
            - scores: Dict[str, float]
                * Dicionário com scores de todas as categorias
                * Útil para debugging e análise detalhada
        
        Exemplo de uso:
        ===============
        >>> analyzer = BERTAnalyzer(sbert_model_name='sentence-transformers/...')
        >>> categories, conf, scores = analyzer.classify_sales_category_multi(
        ...     "Quanto custa e como funciona?",
        ...     min_confidence=0.3,
        ...     similarity_threshold=0.7
        ... )
        >>> print(categories)  # [('price_interest', 0.75), ('information_gathering', 0.68)]
        >>> print(conf)         # 0.82
        >>> print(scores)       # {'price_interest': 0.75, 'information_gathering': 0.68, ...}
        
        Notas:
        ======
        - Este método é opcional e não substitui classify_sales_category()
        - Use quando precisar capturar múltiplas interpretações válidas
        - similarity_threshold controla quão próximos os scores precisam ser
        """
        # Usar método existente para obter scores e métricas
        categoria, confianca, scores, ambiguity, intensity, flags = \
            self.classify_sales_category(text, min_confidence)
        
        if not scores:
            return [], 0.0, {}
        
        # Ordenar categorias por score (maior primeiro)
        sorted_categories = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Se não há categoria detectada (todas abaixo de min_confidence)
        if not categoria:
            # Retornar apenas a melhor categoria mesmo que abaixo do threshold
            # (útil para análise mesmo quando não há confiança suficiente)
            return [sorted_categories[0]] if sorted_categories else [], 0.0, scores
        
        # Determinar quantas categorias incluir
        if len(sorted_categories) < 2:
            # Apenas uma categoria disponível
            return [sorted_categories[0]], confianca, scores
        
        best_score = sorted_categories[0][1]
        second_score = sorted_categories[1][1]
        
        # Se segunda melhor está próxima da melhor, incluir ambas
        # Critério: second_score >= similarity_threshold × best_score
        # E ambas acima de min_confidence
        if second_score >= similarity_threshold * best_score and \
           second_score >= min_confidence:
            # Retornar múltiplas categorias (até max_categories)
            return sorted_categories[:max_categories], confianca, scores
        
        # Caso contrário, apenas a melhor categoria
        return [sorted_categories[0]], confianca, scores
    
    def aggregate_categories_temporal(
        self,
        window: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Agrega categorias em janela temporal para reduzir ruído.
        
        Este método analisa múltiplos chunks em uma janela temporal e calcula:
        - Categoria dominante (mais frequente)
        - Distribuição de categorias
        - Estabilidade da categoria ao longo do tempo
        
        Útil para reduzir falsos positivos de frases isoladas e identificar
        padrões consistentes na conversa.
        
        Args:
        =====
        window: List[Dict[str, Any]]
            Lista de chunks na janela temporal. Cada chunk deve conter:
            - 'sales_category': Optional[str] (categoria detectada)
            - Outros campos opcionais podem estar presentes
        
        Returns:
        ========
        Optional[Dict[str, Any]]
            Dicionário com agregação temporal:
            - 'dominant_category': str (categoria mais frequente)
            - 'category_distribution': Dict[str, float] (distribuição de categorias, 0.0 a 1.0)
            - 'stability': float (0.0 a 1.0, quão estável é a categoria dominante)
            None se window estiver vazio ou não houver categorias válidas
        
        Exemplo de uso:
        ===============
        >>> window = [
        ...     {'sales_category': 'price_interest', 'timestamp': 1000},
        ...     {'sales_category': 'price_interest', 'timestamp': 2000},
        ...     {'sales_category': 'value_exploration', 'timestamp': 3000}
        ... ]
        >>> aggregated = analyzer.aggregate_categories_temporal(window)
        >>> print(aggregated['dominant_category'])  # 'price_interest'
        >>> print(aggregated['stability'])  # ~0.67 (2/3 dos chunks são price_interest)
        """
        if not window:
            return None
        
        # Contar ocorrências de cada categoria
        category_counts: Dict[str, int] = {}
        total_with_category = 0
        
        for chunk in window:
            category = chunk.get('sales_category')
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
                total_with_category += 1
        
        # Se não há categorias válidas, retornar None
        if not category_counts:
            return None
        
        # Calcular distribuição (probabilidades)
        distribution: Dict[str, float] = {
            cat: count / total_with_category
            for cat, count in category_counts.items()
        }
        
        # Encontrar categoria dominante (mais frequente)
        dominant_category = max(category_counts.items(), key=lambda x: x[1])[0]
        
        # Calcular estabilidade
        # Estabilidade = probabilidade da categoria dominante
        # Quanto maior, mais estável é a categoria ao longo do tempo
        stability = distribution[dominant_category]
        
        return {
            'dominant_category': dominant_category,
            'category_distribution': distribution,
            'stability': stability,
            'total_chunks': len(window),
            'chunks_with_category': total_with_category
        }
    
    def detect_category_transition(
        self,
        current_category: Optional[str],
        current_score: float,
        history: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Detecta transições significativas de categoria ao longo do tempo.
        
        Uma transição ocorre quando a categoria atual é diferente da categoria
        anterior e representa uma mudança de estágio na conversa de vendas.
        
        Tipos de transição:
        - 'advancing': Cliente progredindo (ex: value_exploration → price_interest)
        - 'regressing': Cliente regredindo (ex: decision_signal → objection_soft)
        - 'lateral': Mudança sem progressão/regressão clara
        
        Args:
        =====
        current_category: Optional[str]
            Categoria atual detectada
        
        current_score: float
            Score da categoria atual (0.0 a 1.0)
        
        history: List[Dict[str, Any]]
            Histórico de chunks anteriores. Cada chunk deve conter:
            - 'sales_category': Optional[str]
            - 'timestamp': int
        
        Returns:
        ========
        Optional[Dict[str, Any]]
            Dicionário com informações da transição:
            - 'transition_type': str ('advancing' | 'regressing' | 'lateral')
            - 'from_category': str (categoria anterior)
            - 'to_category': str (categoria atual)
            - 'confidence': float (0.0 a 1.0, confiança da transição)
            - 'time_delta_ms': int (tempo decorrido desde última categoria)
            None se não houver transição detectada
        
        Exemplo de uso:
        ===============
        >>> history = [
        ...     {'sales_category': 'value_exploration', 'timestamp': 1000},
        ...     {'sales_category': 'value_exploration', 'timestamp': 2000}
        ... ]
        >>> transition = analyzer.detect_category_transition(
        ...     'price_interest', 0.85, history
        ... )
        >>> print(transition['transition_type'])  # 'advancing'
        >>> print(transition['from_category'])    # 'value_exploration'
        >>> print(transition['to_category'])     # 'price_interest'
        """
        if not current_category or not history:
            return None
        
        # Obter categoria anterior (última com categoria válida)
        previous_category = None
        previous_timestamp = None
        
        # Percorrer histórico de trás para frente para encontrar última categoria válida
        for chunk in reversed(history):
            category = chunk.get('sales_category')
            if category:
                previous_category = category
                previous_timestamp = chunk.get('timestamp')
                break
        
        # Se não há categoria anterior ou é a mesma, não há transição
        if not previous_category or previous_category == current_category:
            return None
        
        # Obter estágios de progressão
        current_stage = CATEGORY_PROGRESSION.get(current_category, 0)
        previous_stage = CATEGORY_PROGRESSION.get(previous_category, 0)
        
        # Determinar tipo de transição baseado na diferença de estágios
        if current_stage > previous_stage:
            transition_type = 'advancing'
        elif current_stage < previous_stage:
            transition_type = 'regressing'
        else:
            transition_type = 'lateral'
        
        # Calcular confiança da transição
        # Baseado na diferença de estágios e no score atual
        stage_diff = abs(current_stage - previous_stage)
        
        # Confiança aumenta com:
        # 1. Maior diferença de estágios (mudança mais significativa)
        # 2. Maior score da categoria atual (mais confiável)
        # Normalizar diferença de estágios (máximo possível é 7: -2 a 5)
        max_stage_diff = 7
        stage_confidence = min(1.0, stage_diff / max_stage_diff)
        
        # Confiança final combina confiança do estágio e score atual
        confidence = stage_confidence * current_score
        
        # Calcular tempo decorrido desde última categoria
        # Usar timestamp do último chunk do histórico como referência atual
        current_timestamp = history[-1].get('timestamp') if history else 0
        time_delta_ms = current_timestamp - previous_timestamp if previous_timestamp else 0
        
        return {
            'transition_type': transition_type,
            'from_category': previous_category,
            'to_category': current_category,
            'confidence': confidence,
            'time_delta_ms': time_delta_ms,
            'from_stage': previous_stage,
            'to_stage': current_stage,
            'stage_difference': stage_diff
        }
    
    def calculate_semantic_trend(
        self,
        window: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calcula tendência semântica da conversa ao longo do tempo.
        
        A tendência indica se a conversa está progredindo, regredindo ou
        permanecendo estável baseado na sequência de categorias na janela temporal.
        
        Args:
        =====
        window: List[Dict[str, Any]]
            Lista de chunks na janela temporal. Cada chunk deve conter:
            - 'sales_category': Optional[str]
            - 'timestamp': int
        
        Returns:
        ========
        Dict[str, Any]
            Dicionário com informações da tendência:
            - 'trend': str ('advancing' | 'stable' | 'regressing')
            - 'trend_strength': float (0.0 a 1.0, força da tendência)
            - 'current_stage': int (estágio atual na progressão)
            - 'velocity': float (velocidade de mudança, positivo = avançando)
        
        Exemplo de uso:
        ===============
        >>> window = [
        ...     {'sales_category': 'value_exploration', 'timestamp': 1000},
        ...     {'sales_category': 'price_interest', 'timestamp': 2000},
        ...     {'sales_category': 'decision_signal', 'timestamp': 3000}
        ... ]
        >>> trend = analyzer.calculate_semantic_trend(window)
        >>> print(trend['trend'])        # 'advancing'
        >>> print(trend['trend_strength'])  # ~0.8
        >>> print(trend['velocity'])     # positivo
        """
        if len(window) < 2:
            return {
                'trend': 'stable',
                'trend_strength': 0.0,
                'current_stage': 0,
                'velocity': 0.0
            }
        
        # Mapear categorias para estágios de progressão
        progression_values: List[int] = []
        for chunk in window:
            category = chunk.get('sales_category')
            if category:
                stage = CATEGORY_PROGRESSION.get(category, 0)
                progression_values.append(stage)
        
        if len(progression_values) < 2:
            # Apenas uma categoria ou nenhuma
            current_stage = progression_values[0] if progression_values else 0
            return {
                'trend': 'stable',
                'trend_strength': 0.0,
                'current_stage': current_stage,
                'velocity': 0.0
            }
        
        # Calcular tendência usando regressão linear simples
        # slope positivo = avançando, negativo = regredindo, próximo de zero = estável
        import numpy as np
        
        # Criar array de índices (posição na sequência)
        x = np.arange(len(progression_values))
        # Array de estágios
        y = np.array(progression_values)
        
        # Calcular slope (coeficiente angular) usando polyfit
        # polyfit retorna [slope, intercept]
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalizar slope para calcular força da tendência
        # Slope máximo esperado seria ~1.0 por chunk (mudança de 1 estágio por chunk)
        # Usar normalização conservadora
        trend_strength = min(1.0, abs(slope) / 2.0)
        
        # Determinar direção da tendência baseado no slope
        if slope > 0.1:
            trend = 'advancing'
        elif slope < -0.1:
            trend = 'regressing'
        else:
            trend = 'stable'
        
        # Estágio atual é o último estágio na sequência
        current_stage = progression_values[-1] if progression_values else 0
        
        return {
            'trend': trend,
            'trend_strength': float(trend_strength),
            'current_stage': current_stage,
            'velocity': float(slope)
        }
    
    def classify_with_context(
        self,
        text: str,
        context_window: List[Dict[str, Any]],
        min_consistency: float = 0.6,
        min_confidence: float = 0.3
    ) -> Dict[str, Any]:
        """
        Classifica texto considerando contexto histórico para reduzir ruído.
        
        Este método usa contexto histórico para validar classificações pontuais.
        Se a categoria atual for inconsistente com o histórico, usa a categoria
        agregada do histórico (mais confiável) ao invés da categoria pontual.
        
        Útil para reduzir falsos positivos de frases isoladas que podem ser
        classificadas incorretamente quando analisadas fora de contexto.
        
        Args:
        =====
        text: str
            Texto atual a ser classificado
        
        context_window: List[Dict[str, Any]]
            Janela de contexto histórico. Cada chunk deve conter:
            - 'sales_category': Optional[str]
            - Outros campos opcionais
        
        min_consistency: float, opcional (padrão: 0.6)
            Consistência mínima necessária para aceitar categoria atual.
            Se consistência < min_consistency e histórico é estável,
            usa categoria histórica ao invés da atual.
        
        min_confidence: float, opcional (padrão: 0.3)
            Score mínimo necessário para aceitar uma classificação
        
        Returns:
        ========
        Dict[str, Any]
            Dicionário com resultado da classificação contextual:
            - 'category': Optional[str] (categoria detectada)
            - 'confidence': float (confiança da classificação)
            - 'is_consistent': bool (se categoria atual é consistente com histórico)
            - 'used_context': bool (True se usou contexto ao invés de chunk atual)
            - 'ambiguity': float (ambiguidade semântica)
            - 'intensity': float (intensidade do sinal)
            - 'flags': Dict[str, bool] (flags semânticas)
        
        Exemplo de uso:
        ===============
        >>> context_window = [
        ...     {'sales_category': 'price_interest', 'timestamp': 1000},
        ...     {'sales_category': 'price_interest', 'timestamp': 2000}
        ... ]
        >>> result = analyzer.classify_with_context(
        ...     "Quanto custa?", context_window
        ... )
        >>> print(result['category'])      # 'price_interest'
        >>> print(result['is_consistent']) # True
        >>> print(result['used_context'])  # False (usou categoria atual)
        """
        # Classificar chunk atual usando método padrão
        categoria, confianca, scores, ambiguity, intensity, flags = \
            self.classify_sales_category(text, min_confidence)
        
        # Se não há contexto histórico, usar classificação atual
        if not context_window:
            return {
                'category': categoria,
                'confidence': confianca,
                'is_consistent': True,
                'used_context': False,
                'ambiguity': ambiguity,
                'intensity': intensity,
                'flags': flags
            }
        
        # Agregar categorias do histórico
        aggregated = self.aggregate_categories_temporal(context_window)
        
        if not aggregated:
            # Histórico não tem categorias válidas, usar classificação atual
            return {
                'category': categoria,
                'confidence': confianca,
                'is_consistent': True,
                'used_context': False,
                'ambiguity': ambiguity,
                'intensity': intensity,
                'flags': flags
            }
        
        dominant_historical = aggregated['dominant_category']
        stability = aggregated['stability']
        
        # Verificar consistência
        # Consistente se:
        # 1. Categoria atual == categoria histórica dominante, OU
        # 2. Histórico é instável (stability < 0.5), então aceitar categoria atual
        is_consistent = (
            categoria == dominant_historical or
            stability < 0.5  # Histórico instável, aceitar atual
        )
        
        # Decidir qual categoria usar
        # Usar categoria atual se:
        # 1. É consistente com histórico, OU
        # 2. Tem confiança muito alta (> 0.8), independente de consistência
        if is_consistent or confianca > 0.8:
            # Usar categoria atual
            return {
                'category': categoria,
                'confidence': confianca,
                'is_consistent': is_consistent,
                'used_context': False,
                'ambiguity': ambiguity,
                'intensity': intensity,
                'flags': flags
            }
        else:
            # Usar categoria histórica (mais confiável)
            # Recalcular flags baseado na categoria histórica
            historical_flags = self._generate_semantic_flags(
                dominant_historical,
                stability,  # Usar estabilidade como confiança
                intensity,  # Manter intensidade atual
                ambiguity   # Manter ambiguidade atual
            )
            
            return {
                'category': dominant_historical,
                'confidence': stability,  # Usar estabilidade como confiança
                'is_consistent': False,
                'used_context': True,
                'ambiguity': ambiguity,
                'intensity': intensity,
                'flags': historical_flags
            }

