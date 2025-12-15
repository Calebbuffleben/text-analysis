"""
Módulo para coleta de métricas semânticas.

Este módulo fornece a classe SemanticMetrics que coleta métricas de qualidade
da classificação semântica de categorias de vendas, permitindo monitoramento
e ajustes contínuos baseados em dados reais.
"""

from typing import Dict, Any, Optional
from collections import defaultdict
import time


class SemanticMetrics:
    """
    Coleta métricas de qualidade da classificação semântica.
    
    Esta classe mantém estatísticas agregadas sobre:
    - Taxa de sucesso de classificações
    - Confiança média, intensidade média, ambiguidade média
    - Distribuição de categorias detectadas
    - Taxa de alta confiança
    - Contagem de transições
    - Taxa de flags ativadas
    
    As métricas são atualizadas usando média móvel exponencial (EMA)
    para dar mais peso a dados recentes.
    
    Exemplo de uso:
    ===============
    >>> metrics = SemanticMetrics()
    >>> metrics.record_classification(
    ...     category='price_interest',
    ...     confidence=0.85,
    ...     intensity=0.92,
    ...     ambiguity=0.15
    ... )
    >>> stats = metrics.get_metrics()
    >>> print(stats['avg_confidence'])  # 0.85
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Inicializa coletor de métricas.
        
        Args:
        =====
        alpha: float, opcional (padrão: 0.1)
            Fator de suavização para média móvel exponencial (EMA).
            Valores menores (ex: 0.05) = mais peso a histórico
            Valores maiores (ex: 0.2) = mais peso a dados recentes
        """
        self.alpha = alpha
        
        # Contadores básicos
        self.total_classifications = 0
        self.successful_classifications = 0
        self.failed_classifications = 0
        
        # Métricas médias (usando EMA)
        self.avg_confidence = 0.0
        self.avg_intensity = 0.0
        self.avg_ambiguity = 0.0
        
        # Distribuição de categorias
        self.category_distribution: Dict[str, int] = defaultdict(int)
        
        # Taxa de alta confiança (classificações com confiança > 0.7)
        self.high_confidence_count = 0
        self.high_confidence_rate = 0.0
        
        # Contadores de flags
        self.flags_count: Dict[str, int] = defaultdict(int)
        
        # Contadores de transições
        self.transitions_count: Dict[str, int] = defaultdict(int)
        
        # Timestamp da última atualização
        self.last_update: Optional[float] = None
    
    def record_classification(
        self,
        category: Optional[str],
        confidence: float,
        intensity: float,
        ambiguity: float,
        flags: Optional[Dict[str, bool]] = None,
        transition: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Registra uma classificação para atualizar métricas.
        
        Args:
        =====
        category: Optional[str]
            Categoria detectada (None se nenhuma)
        
        confidence: float
            Confiança da classificação (0.0 a 1.0)
        
        intensity: float
            Intensidade do sinal (0.0 a 1.0)
        
        ambiguity: float
            Ambiguidade semântica (0.0 a 1.0)
        
        flags: Optional[Dict[str, bool]], opcional
            Flags semânticas ativadas
        
        transition: Optional[Dict[str, Any]], opcional
            Informações sobre transição detectada
        """
        self.total_classifications += 1
        self.last_update = time.time()
        
        # Atualizar contadores de sucesso/falha
        if category:
            self.successful_classifications += 1
            self.category_distribution[category] += 1
        else:
            self.failed_classifications += 1
        
        # Atualizar médias usando EMA (média móvel exponencial)
        # Fórmula: new_avg = alpha * new_value + (1 - alpha) * old_avg
        self.avg_confidence = (
            self.alpha * confidence + (1 - self.alpha) * self.avg_confidence
        )
        self.avg_intensity = (
            self.alpha * intensity + (1 - self.alpha) * self.avg_intensity
        )
        self.avg_ambiguity = (
            self.alpha * ambiguity + (1 - self.alpha) * self.avg_ambiguity
        )
        
        # Atualizar taxa de alta confiança
        if confidence > 0.7:
            self.high_confidence_count += 1
        
        # Calcular taxa de alta confiança (proporção)
        if self.total_classifications > 0:
            self.high_confidence_rate = (
                self.high_confidence_count / self.total_classifications
            )
        
        # Contar flags ativadas
        if flags:
            for flag_name, flag_value in flags.items():
                if flag_value:
                    self.flags_count[flag_name] += 1
        
        # Contar transições
        if transition and transition.get('transition_type'):
            transition_type = transition['transition_type']
            self.transitions_count[transition_type] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas atuais agregadas.
        
        Returns:
        ========
        Dict[str, Any]
            Dicionário com métricas:
            - total_classifications: int
            - successful_classifications: int
            - failed_classifications: int
            - success_rate: float (0.0 a 1.0)
            - avg_confidence: float
            - avg_intensity: float
            - avg_ambiguity: float
            - high_confidence_rate: float (proporção com confiança > 0.7)
            - category_distribution: Dict[str, int]
            - category_distribution_percent: Dict[str, float]
            - flags_count: Dict[str, int]
            - transitions_count: Dict[str, int]
            - last_update: Optional[float] (timestamp)
        """
        # Calcular taxa de sucesso
        success_rate = (
            self.successful_classifications / self.total_classifications
            if self.total_classifications > 0
            else 0.0
        )
        
        # Calcular distribuição percentual de categorias
        category_distribution_percent: Dict[str, float] = {}
        if self.successful_classifications > 0:
            for category, count in self.category_distribution.items():
                category_distribution_percent[category] = (
                    count / self.successful_classifications
                )
        
        return {
            'total_classifications': self.total_classifications,
            'successful_classifications': self.successful_classifications,
            'failed_classifications': self.failed_classifications,
            'success_rate': success_rate,
            'avg_confidence': round(self.avg_confidence, 4),
            'avg_intensity': round(self.avg_intensity, 4),
            'avg_ambiguity': round(self.avg_ambiguity, 4),
            'high_confidence_rate': round(self.high_confidence_rate, 4),
            'category_distribution': dict(self.category_distribution),
            'category_distribution_percent': category_distribution_percent,
            'flags_count': dict(self.flags_count),
            'transitions_count': dict(self.transitions_count),
            'last_update': self.last_update,
            'last_update_iso': (
                time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(self.last_update))
                if self.last_update
                else None
            )
        }
    
    def reset(self) -> None:
        """
        Reseta todas as métricas.
        
        Útil para testes ou quando se deseja começar uma nova coleta.
        """
        self.total_classifications = 0
        self.successful_classifications = 0
        self.failed_classifications = 0
        self.avg_confidence = 0.0
        self.avg_intensity = 0.0
        self.avg_ambiguity = 0.0
        self.category_distribution.clear()
        self.high_confidence_count = 0
        self.high_confidence_rate = 0.0
        self.flags_count.clear()
        self.transitions_count.clear()
        self.last_update = None
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo executivo das métricas.
        
        Útil para dashboards ou relatórios de alto nível.
        
        Returns:
        ========
        Dict[str, Any]
            Resumo com métricas principais
        """
        metrics = self.get_metrics()
        
        # Categoria mais comum
        most_common_category = (
            max(self.category_distribution.items(), key=lambda x: x[1])[0]
            if self.category_distribution
            else None
        )
        
        # Flag mais ativada
        most_common_flag = (
            max(self.flags_count.items(), key=lambda x: x[1])[0]
            if self.flags_count
            else None
        )
        
        # Tipo de transição mais comum
        most_common_transition = (
            max(self.transitions_count.items(), key=lambda x: x[1])[0]
            if self.transitions_count
            else None
        )
        
        return {
            'total_classifications': metrics['total_classifications'],
            'success_rate': metrics['success_rate'],
            'avg_confidence': metrics['avg_confidence'],
            'avg_intensity': metrics['avg_intensity'],
            'avg_ambiguity': metrics['avg_ambiguity'],
            'high_confidence_rate': metrics['high_confidence_rate'],
            'most_common_category': most_common_category,
            'most_common_flag': most_common_flag,
            'most_common_transition': most_common_transition,
            'categories_detected': len(self.category_distribution),
            'flags_activated': sum(self.flags_count.values()),
            'transitions_detected': sum(self.transitions_count.values())
        }

