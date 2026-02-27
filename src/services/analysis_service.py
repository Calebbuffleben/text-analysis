"""
Ponto de entrada público do serviço de análise de texto.
Reexporta TextAnalysisService da arquitetura modular (service_text_analysis).
"""

from .service_text_analysis.analysis.analysis_service import TextAnalysisService

__all__ = ["TextAnalysisService"]
