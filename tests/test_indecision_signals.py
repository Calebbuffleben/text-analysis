"""
Testes unitários para o módulo de cálculo de métricas de indecisão.

Testa a função compute_indecision_metrics_safe do módulo signals.indecision.
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.signals.indecision import compute_indecision_metrics_safe
from src.models.bert_analyzer import BERTAnalyzer


class TestComputeIndecisionMetricsSafe:
    """Testes para compute_indecision_metrics_safe"""
    
    @pytest.fixture
    def mock_analyzer(self):
        """Cria um mock do BERTAnalyzer"""
        analyzer = Mock(spec=BERTAnalyzer)
        return analyzer
    
    def test_gating_sbert_disabled(self, mock_analyzer):
        """sbert_enabled=False → retorna {} e não chama analyzer"""
        result = compute_indecision_metrics_safe(
            analyzer=mock_analyzer,
            sbert_enabled=False,
            sales_category="price_interest",
            sales_category_confidence=0.8,
            sales_category_intensity=0.7,
            sales_category_ambiguity=0.2,
            conditional_keywords_detected=["talvez", "depois"]
        )
        
        assert result == {}
        mock_analyzer.calculate_indecision_metrics.assert_not_called()
    
    def test_gating_category_none(self, mock_analyzer):
        """sales_category=None → retorna {} e não chama analyzer"""
        result = compute_indecision_metrics_safe(
            analyzer=mock_analyzer,
            sbert_enabled=True,
            sales_category=None,
            sales_category_confidence=0.8,
            sales_category_intensity=0.7,
            sales_category_ambiguity=0.2,
            conditional_keywords_detected=["talvez"]
        )
        
        assert result == {}
        mock_analyzer.calculate_indecision_metrics.assert_not_called()
    
    def test_gating_passa_chama_analyzer(self, mock_analyzer):
        """ambos habilitados → chama analyzer.calculate_indecision_metrics()"""
        expected_metrics = {
            'indecision_score': 0.75,
            'postponement_likelihood': 0.65,
            'conditional_language_score': 0.55
        }
        mock_analyzer.calculate_indecision_metrics.return_value = expected_metrics
        
        result = compute_indecision_metrics_safe(
            analyzer=mock_analyzer,
            sbert_enabled=True,
            sales_category="stalling",
            sales_category_confidence=0.8,
            sales_category_intensity=0.7,
            sales_category_ambiguity=0.6,
            conditional_keywords_detected=["talvez", "depois"]
        )
        
        assert result == expected_metrics
        mock_analyzer.calculate_indecision_metrics.assert_called_once()
        
        # Verificar que foi chamado com os parâmetros corretos
        call_args = mock_analyzer.calculate_indecision_metrics.call_args
        assert call_args[0][0] == "stalling"  # sales_category
        assert call_args[0][1] == 0.8  # confidence
        assert call_args[0][2] == 0.7  # intensity
        assert call_args[0][3] == 0.6  # ambiguity
        assert call_args[0][4] == ["talvez", "depois"]  # conditional_keywords
    
    def test_defaults_none_vira_zero(self, mock_analyzer):
        """confidence=None → passa 0.0 para analyzer"""
        expected_metrics = {'indecision_score': 0.5}
        mock_analyzer.calculate_indecision_metrics.return_value = expected_metrics
        
        result = compute_indecision_metrics_safe(
            analyzer=mock_analyzer,
            sbert_enabled=True,
            sales_category="objection_soft",
            sales_category_confidence=None,
            sales_category_intensity=None,
            sales_category_ambiguity=None,
            conditional_keywords_detected=[]
        )
        
        assert result == expected_metrics
        call_args = mock_analyzer.calculate_indecision_metrics.call_args
        assert call_args[0][1] == 0.0  # confidence or 0.0
        assert call_args[0][2] == 0.0  # intensity or 0.0
        assert call_args[0][3] == 0.0  # ambiguity or 0.0
    
    def test_exception_retorna_vazio(self, mock_analyzer):
        """analyzer lança exception → retorna {} e não propaga"""
        mock_analyzer.calculate_indecision_metrics.side_effect = Exception("Erro de teste")
        
        result = compute_indecision_metrics_safe(
            analyzer=mock_analyzer,
            sbert_enabled=True,
            sales_category="price_interest",
            sales_category_confidence=0.8,
            sales_category_intensity=0.7,
            sales_category_ambiguity=0.2,
            conditional_keywords_detected=[]
        )
        
        assert result == {}
        # Não deve propagar a exceção (teste passa se chegar aqui)
    
    def test_retorna_dict_correto(self, mock_analyzer):
        """analyzer retorna dict → função retorna mesmo dict"""
        expected_metrics = {
            'indecision_score': 0.85,
            'postponement_likelihood': 0.75,
            'conditional_language_score': 0.65
        }
        mock_analyzer.calculate_indecision_metrics.return_value = expected_metrics
        
        result = compute_indecision_metrics_safe(
            analyzer=mock_analyzer,
            sbert_enabled=True,
            sales_category="stalling",
            sales_category_confidence=0.9,
            sales_category_intensity=0.8,
            sales_category_ambiguity=0.7,
            conditional_keywords_detected=["pensar", "depois"]
        )
        
        assert result == expected_metrics
        assert isinstance(result, dict)
        assert 'indecision_score' in result
        assert 'postponement_likelihood' in result
        assert 'conditional_language_score' in result
    
    def test_gating_ambos_falsos(self, mock_analyzer):
        """sbert_enabled=False E sales_category=None → retorna {}"""
        result = compute_indecision_metrics_safe(
            analyzer=mock_analyzer,
            sbert_enabled=False,
            sales_category=None,
            sales_category_confidence=0.8,
            sales_category_intensity=0.7,
            sales_category_ambiguity=0.2,
            conditional_keywords_detected=[]
        )
        
        assert result == {}
        mock_analyzer.calculate_indecision_metrics.assert_not_called()

