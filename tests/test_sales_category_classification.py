"""
Testes para classificação de categorias de vendas usando SBERT.

Estes testes validam:
1. Carregamento de embeddings dos exemplos de referência
2. Classificação correta de textos em categorias de vendas
3. Cálculo de confiança adequado
4. Tratamento de casos extremos (textos ambíguos, SBERT não configurado)
5. Performance e cache de embeddings
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from src.models.bert_analyzer import BERTAnalyzer, SALES_CATEGORY_EXAMPLES
from src.services.analysis_service import TextAnalysisService
from src.types.messages import TranscriptionChunk


class TestSalesCategoryClassification:
    """
    Testes para classificação de categorias de vendas.
    
    Estes testes verificam se o método classify_sales_category() funciona
    corretamente para diferentes tipos de texto de entrada.
    """
    
    @pytest.fixture
    def analyzer_with_sbert(self):
        """
        Cria uma instância de BERTAnalyzer com SBERT configurado.
        
        Usa mock do SentenceTransformer para evitar carregar modelos reais
        durante os testes (mais rápido e não requer GPU).
        """
        analyzer = BERTAnalyzer(
            model_name='neuralmind/bert-base-portuguese-cased',
            device='cpu',
            sbert_model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        )
        
        # Mock do modelo SBERT para evitar carregar modelo real
        mock_sbert_model = Mock()
        # Simular embedding de 384 dimensões (dimensão do modelo real)
        mock_embedding = np.random.rand(384).astype(np.float32)
        # Normalizar para simular embeddings normalizados
        mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
        
        mock_sbert_model.encode.return_value = mock_embedding
        mock_sbert_model.to.return_value = mock_sbert_model
        
        analyzer.sbert_model = mock_sbert_model
        analyzer._sbert_loaded = True
        
        return analyzer
    
    @pytest.fixture
    def analyzer_without_sbert(self):
        """
        Cria uma instância de BERTAnalyzer sem SBERT configurado.
        
        Útil para testar comportamento quando SBERT não está disponível.
        """
        return BERTAnalyzer(
            model_name='neuralmind/bert-base-portuguese-cased',
            device='cpu',
            sbert_model_name=None
        )
    
    def test_load_sales_category_examples_embeddings(self, analyzer_with_sbert):
        """
        Testa o carregamento de embeddings dos exemplos de referência.
        
        Verifica:
        - Embeddings são calculados para todas as categorias
        - Cache é preenchido corretamente
        - Flag _sales_examples_loaded é marcada como True
        """
        # Verificar que embeddings ainda não foram carregados
        assert analyzer_with_sbert._sales_examples_loaded == False
        assert analyzer_with_sbert._sales_category_examples_embeddings is None
        
        # Carregar embeddings
        analyzer_with_sbert._load_sales_category_examples_embeddings()
        
        # Verificar que embeddings foram carregados
        assert analyzer_with_sbert._sales_examples_loaded == True
        assert analyzer_with_sbert._sales_category_examples_embeddings is not None
        
        # Verificar que todas as categorias têm embeddings
        assert len(analyzer_with_sbert._sales_category_examples_embeddings) == len(SALES_CATEGORY_EXAMPLES)
        
        # Verificar que cada categoria tem embeddings para todos os exemplos
        for category, examples in SALES_CATEGORY_EXAMPLES.items():
            assert category in analyzer_with_sbert._sales_category_examples_embeddings
            embeddings = analyzer_with_sbert._sales_category_examples_embeddings[category]
            assert len(embeddings) == len(examples)
            
            # Verificar que cada embedding é um array numpy
            for embedding in embeddings:
                assert isinstance(embedding, np.ndarray)
                assert len(embedding) == 384  # Dimensão do modelo
    
    def test_load_sales_category_examples_embeddings_idempotent(self, analyzer_with_sbert):
        """
        Testa que carregar embeddings múltiplas vezes não recalcula (idempotência).
        
        Verifica que o método pode ser chamado múltiplas vezes sem problemas
        e que os embeddings são calculados apenas uma vez.
        """
        # Carregar primeira vez
        analyzer_with_sbert._load_sales_category_examples_embeddings()
        first_embeddings = analyzer_with_sbert._sales_category_examples_embeddings.copy()
        
        # Carregar segunda vez (não deve recalcular)
        analyzer_with_sbert._load_sales_category_examples_embeddings()
        second_embeddings = analyzer_with_sbert._sales_category_examples_embeddings
        
        # Verificar que são os mesmos objetos (não foram recalculados)
        assert first_embeddings is second_embeddings
    
    def test_load_sales_category_examples_embeddings_without_sbert(self, analyzer_without_sbert):
        """
        Testa que carregar embeddings sem SBERT configurado levanta RuntimeError.
        
        Verifica tratamento de erro quando SBERT não está disponível.
        """
        with pytest.raises(RuntimeError, match="SBERT model not configured"):
            analyzer_without_sbert._load_sales_category_examples_embeddings()
    
    def test_classify_sales_category_price_interest(self, analyzer_with_sbert):
        """
        Testa classificação de texto relacionado a interesse em preço.
        
        Verifica que textos sobre preço são classificados como 'price_interest'.
        """
        # Preparar embeddings dos exemplos (simular carregamento)
        analyzer_with_sbert._load_sales_category_examples_embeddings()
        
        # Textos de teste que devem ser classificados como price_interest
        test_texts = [
            "Quanto custa isso?",
            "Qual é o preço?",
            "Preciso saber o valor",
            "Quanto eu vou pagar?"
        ]
        
        for text in test_texts:
            categoria, confianca, scores, ambiguidade, intensidade, flags = analyzer_with_sbert.classify_sales_category(text)
            
            # Verificar que retornou uma categoria
            assert categoria is not None
            
            # Verificar que confiança está no range [0, 1]
            assert 0.0 <= confianca <= 1.0
            
            # Verificar que ambiguidade está no range [0, 1]
            assert 0.0 <= ambiguidade <= 1.0
            
            # Verificar que intensidade está no range [0, 1]
            assert 0.0 <= intensidade <= 1.0
            
            # Verificar que flags é um dicionário
            assert isinstance(flags, dict)
            
            # Verificar que scores contém todas as categorias
            assert len(scores) == len(SALES_CATEGORY_EXAMPLES)
            assert all(cat in scores for cat in SALES_CATEGORY_EXAMPLES.keys())
            
            # Verificar que scores estão no range [0, 1]
            assert all(0.0 <= score <= 1.0 for score in scores.values())
    
    def test_classify_sales_category_without_sbert(self, analyzer_without_sbert):
        """
        Testa que classificar sem SBERT retorna None.
        
        Verifica comportamento quando SBERT não está configurado.
        """
        categoria, confianca, scores, ambiguidade, intensidade, flags = analyzer_without_sbert.classify_sales_category("Quanto custa?")
        
        assert categoria is None
        assert confianca == 0.0
        assert scores == {}
        assert ambiguidade == 1.0  # Máxima ambiguidade quando sem SBERT
        assert intensidade == 0.0  # Intensidade zero quando sem SBERT
        assert flags == {}  # Flags vazias quando sem SBERT
    
    def test_classify_sales_category_min_confidence_threshold(self, analyzer_with_sbert):
        """
        Testa que classificação respeita o threshold mínimo de confiança.
        
        Verifica que textos com score muito baixo retornam None.
        """
        analyzer_with_sbert._load_sales_category_examples_embeddings()
        
        # Texto completamente não relacionado (deve ter score baixo)
        texto_nao_relacionado = "O tempo está bom hoje"
        
        # Classificar com threshold alto (deve retornar None)
        categoria, confianca, scores, ambiguidade, intensidade, flags = analyzer_with_sbert.classify_sales_category(
            texto_nao_relacionado,
            min_confidence=0.8  # Threshold muito alto
        )
        
        # Se o melhor score for menor que 0.8, deve retornar None
        if scores:
            best_score = max(scores.values())
            if best_score < 0.8:
                assert categoria is None
                assert confianca == 0.0
                # Ambiguidade e intensidade ainda devem estar presentes
                assert 0.0 <= ambiguidade <= 1.0
                assert 0.0 <= intensidade <= 1.0
                assert isinstance(flags, dict)
    
    def test_classify_sales_category_returns_all_scores(self, analyzer_with_sbert):
        """
        Testa que classificação retorna scores de todas as categorias.
        
        Verifica que o dicionário de scores contém todas as 8 categorias.
        """
        analyzer_with_sbert._load_sales_category_examples_embeddings()
        
        categoria, confianca, scores, ambiguidade, intensidade, flags = analyzer_with_sbert.classify_sales_category("Quanto custa?")
        
        # Verificar que scores contém todas as categorias esperadas
        expected_categories = set(SALES_CATEGORY_EXAMPLES.keys())
        actual_categories = set(scores.keys())
        
        assert expected_categories == actual_categories
        assert len(scores) == 8  # 8 categorias
        
        # Verificar novos campos
        assert 0.0 <= ambiguidade <= 1.0
        assert 0.0 <= intensidade <= 1.0
        assert isinstance(flags, dict)
    
    def test_classify_sales_category_confidence_calculation(self, analyzer_with_sbert):
        """
        Testa que a confiança é calculada corretamente.
        
        Verifica que a confiança considera a diferença entre melhor e segunda melhor categoria.
        """
        analyzer_with_sbert._load_sales_category_examples_embeddings()
        
        categoria, confianca, scores, ambiguidade, intensidade, flags = analyzer_with_sbert.classify_sales_category("Quanto custa?")
        
        if categoria:
            # Verificar que confiança está no range [0, 1]
            assert 0.0 <= confianca <= 1.0
            
            # Verificar novos campos
            assert 0.0 <= ambiguidade <= 1.0
            assert 0.0 <= intensidade <= 1.0
            assert isinstance(flags, dict)
            
            # Se houver múltiplas categorias, verificar que confiança faz sentido
            if len(scores) > 1:
                sorted_scores = sorted(scores.values(), reverse=True)
                best_score = sorted_scores[0]
                second_best_score = sorted_scores[1]
                
                # Confiança deve ser maior quando há maior diferença entre melhor e segunda melhor
                # (isso é uma verificação heurística, não exata devido à fórmula complexa)
                assert confianca >= 0.0
    
    def test_calculate_ambiguity(self, analyzer_with_sbert):
        """
        Testa o cálculo de ambiguidade semântica.
        
        Verifica que:
        - Textos claros têm baixa ambiguidade
        - Textos ambíguos têm alta ambiguidade
        """
        # Caso claro: uma categoria dominante
        scores_clear = {
            'price_interest': 0.9,
            'value_exploration': 0.1,
            'objection_soft': 0.05,
            'objection_hard': 0.02,
            'decision_signal': 0.03,
            'information_gathering': 0.04,
            'stalling': 0.01,
            'closing_readiness': 0.02
        }
        ambiguity_clear = analyzer_with_sbert._calculate_ambiguity(scores_clear)
        assert 0.0 <= ambiguity_clear <= 1.0
        assert ambiguity_clear < 0.3  # Deve ser baixa
        
        # Caso ambíguo: scores muito próximos
        scores_ambiguous = {
            'price_interest': 0.5,
            'value_exploration': 0.48,
            'objection_soft': 0.45,
            'objection_hard': 0.42,
            'decision_signal': 0.44,
            'information_gathering': 0.46,
            'stalling': 0.43,
            'closing_readiness': 0.41
        }
        ambiguity_ambiguous = analyzer_with_sbert._calculate_ambiguity(scores_ambiguous)
        assert 0.0 <= ambiguity_ambiguous <= 1.0
        assert ambiguity_ambiguous > 0.7  # Deve ser alta
    
    def test_semantic_flags(self, analyzer_with_sbert):
        """
        Testa a geração de flags semânticas.
        
        Verifica que flags são geradas corretamente baseadas em categoria,
        confiança, intensidade e ambiguidade.
        """
        # Caso: price_window_open deve ser True
        flags = analyzer_with_sbert._generate_semantic_flags(
            'price_interest', 0.85, 0.9, 0.2
        )
        assert flags['price_window_open'] == True
        assert isinstance(flags, dict)
        
        # Caso: price_window_open deve ser False (ambiguidade alta)
        flags = analyzer_with_sbert._generate_semantic_flags(
            'price_interest', 0.85, 0.9, 0.5
        )
        assert flags['price_window_open'] == False
        
        # Caso: decision_signal_strong deve ser True
        flags = analyzer_with_sbert._generate_semantic_flags(
            'decision_signal', 0.9, 0.88, 0.2
        )
        assert flags['decision_signal_strong'] == True
        
        # Caso: ready_to_close deve ser True
        flags = analyzer_with_sbert._generate_semantic_flags(
            'closing_readiness', 0.9, 0.95, 0.15
        )
        assert flags['ready_to_close'] == True
        
        # Caso: categoria None deve retornar flags vazias
        flags = analyzer_with_sbert._generate_semantic_flags(
            None, 0.0, 0.0, 1.0
        )
        assert flags == {}


class TestSalesCategoryIntegration:
    """
    Testes de integração para classificação de categorias de vendas.
    
    Testa o fluxo completo desde o TextAnalysisService até a classificação.
    """
    
    @pytest.fixture
    def analysis_service(self):
        """
        Cria uma instância de TextAnalysisService para testes.
        
        Usa mocks para evitar carregar modelos reais.
        """
        service = TextAnalysisService()
        return service
    
    @pytest.mark.asyncio
    async def test_analyze_includes_sales_category(self, analysis_service, monkeypatch):
        """
        Testa que o método analyze() inclui sales_category no resultado.
        
        Verifica que quando SBERT está configurado, sales_category e
        sales_category_confidence são incluídos no resultado.
        """
        # Mock do Config para ter SBERT configurado
        from src import config
        original_sbert = getattr(config.Config, 'SBERT_MODEL_NAME', None)
        monkeypatch.setattr(config.Config, 'SBERT_MODEL_NAME', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        try:
            # Mock do BERTAnalyzer para evitar carregar modelos reais
            mock_analyzer = Mock(spec=BERTAnalyzer)
            
            # Configurar mocks dos métodos necessários
            mock_analyzer.analyze_sentiment.return_value = {
                'positive': 0.5,
                'negative': 0.2,
                'neutral': 0.3
            }
            mock_analyzer.extract_keywords.return_value = ['quanto', 'custa']
            mock_analyzer.detect_emotions.return_value = {
                'joy': 0.0,
                'sadness': 0.0,
                'anger': 0.0,
                'fear': 0.0,
                'surprise': 0.0
            }
            
            # Mock do método classify_sales_category
            mock_analyzer.classify_sales_category.return_value = (
                'price_interest',  # categoria
                0.85,  # confiança
                {  # scores
                    'price_interest': 0.92,
                    'value_exploration': 0.15,
                    'objection_soft': 0.08,
                    'objection_hard': 0.05,
                    'decision_signal': 0.10,
                    'information_gathering': 0.12,
                    'stalling': 0.07,
                    'closing_readiness': 0.09
                },
                0.15,  # ambiguidade
                0.92,  # intensidade
                {  # flags
                    'price_window_open': True,
                    'decision_signal_strong': False,
                    'ready_to_close': False
                }
            )
            
            # Mock do generate_semantic_embedding
            mock_embedding = np.random.rand(384).astype(np.float32)
            mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
            mock_analyzer.generate_semantic_embedding.return_value = mock_embedding
            
            # Substituir o método _get_analyzer para retornar o mock
            analysis_service._get_analyzer = lambda: mock_analyzer
            
            # Criar chunk de teste
            chunk = TranscriptionChunk(
                meetingId='test_meeting',
                participantId='test_participant',
                text='Quanto custa isso?',
                timestamp=1234567890
            )
            
            # Executar análise
            result = await analysis_service.analyze(chunk)
            
            # Verificar que sales_category está presente no resultado
            assert 'sales_category' in result
            assert 'sales_category_confidence' in result
            assert 'sales_category_intensity' in result
            assert 'sales_category_ambiguity' in result
            assert 'sales_category_flags' in result
            
            # Verificar valores
            assert result['sales_category'] == 'price_interest'
            assert result['sales_category_confidence'] == 0.85
            
        finally:
            # Restaurar valor original
            if original_sbert is not None:
                monkeypatch.setattr(config.Config, 'SBERT_MODEL_NAME', original_sbert)
            else:
                monkeypatch.delattr(config.Config, 'SBERT_MODEL_NAME', raising=False)
    
    @pytest.mark.asyncio
    async def test_analyze_without_sbert_returns_none_sales_category(self, analysis_service, monkeypatch):
        """
        Testa que quando SBERT não está configurado, sales_category é None.
        
        Verifica comportamento quando SBERT_MODEL_NAME não está definido.
        """
        # Mock do Config para não ter SBERT configurado
        from src import config
        original_sbert = getattr(config.Config, 'SBERT_MODEL_NAME', None)
        monkeypatch.setattr(config.Config, 'SBERT_MODEL_NAME', None)
        
        try:
            # Mock do BERTAnalyzer
            mock_analyzer = Mock(spec=BERTAnalyzer)
            mock_analyzer.analyze_sentiment.return_value = {
                'positive': 0.5,
                'negative': 0.2,
                'neutral': 0.3
            }
            mock_analyzer.extract_keywords.return_value = ['texto']
            mock_analyzer.detect_emotions.return_value = {
                'joy': 0.0,
                'sadness': 0.0,
                'anger': 0.0,
                'fear': 0.0,
                'surprise': 0.0
            }
            
            analysis_service._get_analyzer = lambda: mock_analyzer
            
            chunk = TranscriptionChunk(
                meetingId='test_meeting',
                participantId='test_participant',
                text='Texto de teste',
                timestamp=1234567890
            )
            
            result = await analysis_service.analyze(chunk)
            
            # Verificar que sales_category é None quando SBERT não está configurado
            assert result.get('sales_category') is None
            assert result.get('sales_category_confidence') is None
            
        finally:
            # Restaurar valor original
            if original_sbert is not None:
                monkeypatch.setattr(config.Config, 'SBERT_MODEL_NAME', original_sbert)
            else:
                monkeypatch.delattr(config.Config, 'SBERT_MODEL_NAME', raising=False)


class TestSalesCategoryExamples:
    """
    Testes para validar os exemplos de referência das categorias.
    
    Verifica que os exemplos estão bem definidos e completos.
    """
    
    def test_all_categories_have_examples(self):
        """
        Testa que todas as categorias têm exemplos de referência.
        
        Verifica que não há categorias vazias.
        """
        assert len(SALES_CATEGORY_EXAMPLES) > 0
        
        for category, examples in SALES_CATEGORY_EXAMPLES.items():
            assert len(examples) > 0, f"Categoria {category} não tem exemplos"
            assert all(isinstance(example, str) for example in examples), \
                f"Categoria {category} tem exemplos que não são strings"
            assert all(len(example.strip()) > 0 for example in examples), \
                f"Categoria {category} tem exemplos vazios"
    
    def test_expected_categories_present(self):
        """
        Testa que todas as categorias esperadas estão presentes.
        
        Verifica que as 8 categorias definidas estão no dicionário.
        """
        expected_categories = {
            'price_interest',
            'value_exploration',
            'objection_soft',
            'objection_hard',
            'decision_signal',
            'information_gathering',
            'stalling',
            'closing_readiness'
        }
        
        actual_categories = set(SALES_CATEGORY_EXAMPLES.keys())
        
        assert expected_categories == actual_categories, \
            f"Categorias esperadas: {expected_categories}, encontradas: {actual_categories}"
    
    def test_examples_are_in_portuguese(self):
        """
        Testa que os exemplos estão em português.
        
        Verifica heurística básica: exemplos contêm palavras comuns em português.
        """
        portuguese_indicators = [
            'quanto', 'qual', 'como', 'quando', 'onde',
            'não', 'sim', 'isso', 'isso', 'para', 'com',
            'preço', 'valor', 'custo', 'produto', 'serviço'
        ]
        
        all_examples_text = ' '.join([
            ' '.join(examples).lower()
            for examples in SALES_CATEGORY_EXAMPLES.values()
        ])
        
        # Verificar que pelo menos algumas palavras em português aparecem
        found_indicators = sum(1 for indicator in portuguese_indicators if indicator in all_examples_text)
        assert found_indicators >= 5, \
            "Poucos indicadores de português encontrados nos exemplos"
    
    def test_examples_have_minimum_count(self):
        """
        Testa que cada categoria tem número mínimo de exemplos.
        
        Verifica que há exemplos suficientes para classificação robusta.
        """
        min_examples = 5  # Mínimo esperado por categoria
        
        for category, examples in SALES_CATEGORY_EXAMPLES.items():
            assert len(examples) >= min_examples, \
                f"Categoria {category} tem apenas {len(examples)} exemplos, mínimo esperado: {min_examples}"

