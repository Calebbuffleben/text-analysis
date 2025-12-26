"""
Módulo de cálculo de métricas de indecisão do cliente.

Este módulo encapsula a lógica de orquestração para cálculo de métricas
de indecisão, incluindo gating (condições para calcular) e tratamento
de erros seguro.
"""

from typing import Dict, Any, List, Optional
from ..models.bert_analyzer import BERTAnalyzer
import structlog

logger = structlog.get_logger()


def compute_indecision_metrics_safe(
    analyzer: BERTAnalyzer,
    sbert_enabled: bool,
    sales_category: Optional[str],
    sales_category_confidence: Optional[float],
    sales_category_intensity: Optional[float],
    sales_category_ambiguity: Optional[float],
    conditional_keywords_detected: List[str],
    meeting_id: str = ""
) -> Dict[str, Any]:
    """
    Calcula métricas de indecisão de forma segura (não quebra o fluxo se falhar).
    
    Esta função encapsula a lógica de orquestração para cálculo de métricas
    de indecisão, incluindo gating (condições para calcular) e tratamento
    de erros. Retorna um dict vazio se não calcular ou se falhar.
    
    Args:
        analyzer: Instância de BERTAnalyzer para calcular métricas
        sbert_enabled: Se SBERT está habilitado (bool(Config.SBERT_MODEL_NAME))
        sales_category: Categoria de vendas detectada (None se nenhuma)
        sales_category_confidence: Confiança da classificação (None ou float)
        sales_category_intensity: Intensidade do sinal semântico (None ou float)
        sales_category_ambiguity: Ambiguidade semântica (None ou float)
        conditional_keywords_detected: Lista de keywords condicionais detectadas
        meeting_id: ID da reunião (para logs, se necessário)
    
    Returns:
        Dict[str, Any] com métricas de indecisão ou dict vazio {} se não calcular/falhar.
        Métricas possíveis: indecision_score, postponement_likelihood, conditional_language_score
    """
    indecision_metrics: Dict[str, Any] = {}
    try:
        if sbert_enabled and sales_category is not None:
            indecision_metrics = analyzer.calculate_indecision_metrics(
                sales_category,
                sales_category_confidence or 0.0,
                sales_category_intensity or 0.0,
                sales_category_ambiguity or 0.0,
                conditional_keywords_detected
            )
    except Exception as e:
        # Não bloquear análise se cálculo de métricas falhar
        # Retorna {} (dict vazio) para indicar que não foi possível calcular
        pass
    
    return indecision_metrics

