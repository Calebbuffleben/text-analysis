"""
Módulo para gerenciar contexto conversacional em análises de texto.

Este módulo fornece a classe ConversationContext que mantém histórico semântico
da conversa, permitindo análise contextual baseada em janelas temporais.
"""

from typing import Dict, List, Optional, Any


class ConversationContext:
    """
    Mantém contexto semântico da conversa para análise temporal.
    
    Esta classe armazena histórico de chunks de texto analisados, permitindo
    análise contextual baseada em janelas temporais. Útil para:
    - Agregação temporal de categorias
    - Detecção de transições de estágio
    - Cálculo de tendências semânticas
    - Redução de ruído de frases isoladas
    
    A classe mantém duas janelas:
    - Janela por tamanho: últimos N chunks (padrão: 10)
    - Janela por tempo: últimos N milissegundos (padrão: 60000ms = 60s)
    
    Exemplo de uso:
    ===============
    >>> ctx = ConversationContext(window_size=10, window_duration_ms=60000)
    >>> ctx.add_chunk({
    ...     'text': 'Quanto custa?',
    ...     'sales_category': 'price_interest',
    ...     'sales_category_confidence': 0.85,
    ...     'timestamp': 1234567890
    ... })
    >>> window = ctx.get_window(1234567890)
    >>> len(window)  # 1
    """
    
    def __init__(self, window_size: int = 10, window_duration_ms: int = 60000):
        """
        Inicializa contexto conversacional.
        
        Args:
        =====
        window_size: int, opcional (padrão: 10)
            Número máximo de chunks a manter na janela por tamanho
        
        window_duration_ms: int, opcional (padrão: 60000)
            Duração da janela temporal em milissegundos (padrão: 60 segundos)
        """
        self.window_size = window_size
        self.window_duration_ms = window_duration_ms
        self.history: List[Dict[str, Any]] = []
    
    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        """
        Adiciona chunk ao histórico e remove chunks antigos se necessário.
        
        O chunk deve conter pelo menos:
        - 'text': str (texto do chunk)
        - 'timestamp': int (timestamp em milissegundos)
        
        Campos opcionais que podem ser incluídos:
        - 'sales_category': str
        - 'sales_category_confidence': float
        - 'sales_category_intensity': float
        - 'sales_category_ambiguity': float
        - 'embedding': List[float]
        
        Args:
        =====
        chunk: Dict[str, Any]
            Dicionário com dados do chunk a ser adicionado
        
        Exemplo:
        ========
        >>> ctx.add_chunk({
        ...     'text': 'Quanto custa?',
        ...     'sales_category': 'price_interest',
        ...     'sales_category_confidence': 0.85,
        ...     'timestamp': 1234567890
        ... })
        """
        # Criar entrada padronizada no histórico
        history_entry = {
            'text': chunk.get('text', ''),
            'sales_category': chunk.get('sales_category'),
            'sales_category_confidence': chunk.get('sales_category_confidence'),
            'sales_category_intensity': chunk.get('sales_category_intensity'),
            'sales_category_ambiguity': chunk.get('sales_category_ambiguity'),
            'timestamp': chunk.get('timestamp', 0),
            'embedding': chunk.get('embedding')
        }
        
        self.history.append(history_entry)
        self._prune_history()
    
    def _prune_history(self) -> None:
        """
        Remove chunks antigos da janela baseado em tamanho e tempo.
        
        Este método é chamado automaticamente após adicionar um chunk.
        Remove chunks que estão fora da janela temporal ou que excedem
        o tamanho máximo da janela.
        """
        if not self.history:
            return
        
        # Remover por tamanho (manter apenas últimos N chunks)
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]
        
        # Remover por tempo (manter apenas chunks dentro da janela temporal)
        if self.history:
            # Usar timestamp do último chunk como referência
            now = self.history[-1]['timestamp']
            cutoff = now - self.window_duration_ms
            
            # Filtrar chunks dentro da janela temporal
            self.history = [
                chunk for chunk in self.history
                if chunk['timestamp'] >= cutoff
            ]
    
    def get_window(self, now: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retorna chunks na janela temporal atual.
        
        A janela inclui chunks que estão dentro do intervalo temporal
        [now - window_duration_ms, now] e respeita o limite de tamanho.
        
        Args:
        =====
        now: Optional[int], opcional
            Timestamp atual em milissegundos.
            Se None, usa o timestamp do último chunk adicionado.
        
        Returns:
        ========
        List[Dict[str, Any]]
            Lista de chunks na janela temporal, ordenados por timestamp
            (mais antigo primeiro). Cada chunk contém:
            - 'text': str
            - 'sales_category': Optional[str]
            - 'sales_category_confidence': Optional[float]
            - 'sales_category_intensity': Optional[float]
            - 'sales_category_ambiguity': Optional[float]
            - 'timestamp': int
            - 'embedding': Optional[List[float]]
        
        Exemplo:
        ========
        >>> window = ctx.get_window(1234567890)
        >>> len(window)  # Número de chunks na janela
        >>> window[0]['sales_category']  # Categoria do chunk mais antigo
        """
        if not self.history:
            return []
        
        # Se now não foi fornecido, usar timestamp do último chunk
        if now is None:
            now = self.history[-1]['timestamp'] if self.history else 0
        
        # Calcular cutoff temporal
        cutoff = now - self.window_duration_ms
        
        # Filtrar chunks dentro da janela temporal
        window = [
            chunk for chunk in self.history
            if chunk['timestamp'] >= cutoff
        ]
        
        # Limitar pelo tamanho da janela (últimos N chunks)
        return window[-self.window_size:]
    
    def clear(self) -> None:
        """
        Limpa todo o histórico.
        
        Útil para testes ou quando se deseja resetar o contexto.
        
        Exemplo:
        ========
        >>> ctx.clear()
        >>> len(ctx.history)  # 0
        """
        self.history = []
    
    def get_history_size(self) -> int:
        """
        Retorna o tamanho atual do histórico.
        
        Returns:
        ========
        int: Número de chunks no histórico
        """
        return len(self.history)
    
    def get_window_info(self, now: Optional[int] = None) -> Dict[str, Any]:
        """
        Retorna informações sobre a janela atual.
        
        Útil para debugging e monitoramento.
        
        Args:
        =====
        now: Optional[int], opcional
            Timestamp atual em milissegundos
        
        Returns:
        ========
        Dict[str, Any]
            Dicionário com informações da janela:
            - 'window_size': int (tamanho atual da janela)
            - 'history_size': int (tamanho total do histórico)
            - 'window_duration_ms': int (duração da janela)
            - 'oldest_timestamp': Optional[int] (timestamp do chunk mais antigo)
            - 'newest_timestamp': Optional[int] (timestamp do chunk mais novo)
        """
        window = self.get_window(now)
        
        info = {
            'window_size': len(window),
            'history_size': len(self.history),
            'window_duration_ms': self.window_duration_ms,
            'oldest_timestamp': window[0]['timestamp'] if window else None,
            'newest_timestamp': window[-1]['timestamp'] if window else None
        }
        
        return info

