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

