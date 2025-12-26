"""
Módulo de detecção de marcadores de reformulação/teach-back.

Este módulo detecta quando o cliente reformula ou repete informações
que foram apresentadas, indicando que ele está processando e tentando
compreender a solução ("solução foi compreendida").
"""

from typing import List, Dict


def detect_reformulation_markers(text: str) -> List[str]:
    """
    Detecta marcadores de reformulação/teach-back (PT-BR) no texto.

    Retorna uma lista de marcadores encontrados (strings).
    """
    t = (text or "").lower()
    markers = [
        "deixa eu ver se entendi",
        "só pra confirmar",
        "se eu entendi",
        "entendi então",
        "entendi que",
        "então vocês",
        "então o que você está dizendo é",
        "quer dizer que",
        "ou seja",
        "resumindo",
        "em resumo",
        "na prática então",
        "basicamente",
    ]
    found: List[str] = []
    for m in markers:
        if m in t:
            found.append(m)
    return found


def compute_reformulation_marker_score(markers: List[str]) -> float:
    """
    Calcula o score de reformulação baseado no número de marcadores encontrados.
    
    Args:
        markers: Lista de marcadores de reformulação detectados
    
    Returns:
        Score entre 0.0 e 1.0, calculado como min(1.0, len(markers) / 2.0)
    """
    return min(1.0, len(markers) / 2.0)


def apply_solution_reformulation_signal_flag(
    flags: Dict[str, bool],
    marker_score: float
) -> None:
    """
    Aplica a flag 'solution_reformulation_signal' no dicionário de flags.
    
    Esta função aplica o efeito colateral de setar a flag quando há
    marcadores de reformulação detectados (score > 0.0).
    
    Args:
        flags: Dicionário de flags de sales_category onde a flag será aplicada
        marker_score: Score de marcadores de reformulação calculado
    
    Nota:
        Não cria a key se score <= 0.0 (não seta False, não cria key vazia).
    """
    if marker_score > 0.0:
        flags['solution_reformulation_signal'] = True

