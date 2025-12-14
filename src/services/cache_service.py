"""
Serviço de cache em memória para resultados de análise de texto.
Utiliza TTLCache do cachetools para cache thread-safe com TTL automático.
"""

from typing import Dict, Any, Optional
from cachetools import TTLCache
import structlog
import hashlib

logger = structlog.get_logger()


class AnalysisCache:
    """
    Cache em memória com TTL para resultados de análise de texto.
    
    Características:
    - Thread-safe (TTLCache é thread-safe)
    - TTL automático (entradas expiram após TTL)
    - Tamanho limitado (evita memory leak)
    - Hash de chaves para evitar colisões
    """
    
    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        """
        Inicializa cache.
        
        Args:
            ttl_seconds: Tempo de vida das entradas em segundos
            max_size: Número máximo de entradas no cache
        """
        self.cache = TTLCache(maxsize=max_size, ttl=ttl_seconds)
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        
        logger.info(
            "✅ [CACHE] AnalysisCache inicializado",
            ttl_seconds=ttl_seconds,
            max_size=max_size
        )
    
    def _generate_key(self, meeting_id: str, participant_id: str, text: str) -> str:
        """
        Gera chave de cache baseada em hash do texto.
        
        Args:
            meeting_id: ID da reunião
            participant_id: ID do participante
            text: Texto a ser analisado
            
        Returns:
            Chave de cache única
        """
        # Usar hash MD5 para textos longos, texto completo para textos curtos
        if len(text) > 100:
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            return f"{meeting_id}:{participant_id}:{text_hash}"
        else:
            return f"{meeting_id}:{participant_id}:{text[:100]}"
    
    def get(self, meeting_id: str, participant_id: str, text: str) -> Optional[Dict[str, Any]]:
        """
        Recupera resultado do cache.
        
        Args:
            meeting_id: ID da reunião
            participant_id: ID do participante
            text: Texto analisado
            
        Returns:
            Resultado da análise ou None se não encontrado/expirado
        """
        key = self._generate_key(meeting_id, participant_id, text)
        result = self.cache.get(key)
        
        if result:
            logger.debug(
                "Cache hit",
                meeting_id=meeting_id,
                key_preview=key[:50]
            )
        else:
            logger.debug(
                "Cache miss",
                meeting_id=meeting_id,
                key_preview=key[:50]
            )
        
        return result
    
    def set(self, meeting_id: str, participant_id: str, text: str, value: Dict[str, Any]):
        """
        Armazena resultado no cache.
        
        Args:
            meeting_id: ID da reunião
            participant_id: ID do participante
            text: Texto analisado
            value: Resultado da análise
        """
        key = self._generate_key(meeting_id, participant_id, text)
        self.cache[key] = value
        
        logger.debug(
            "Cache set",
            meeting_id=meeting_id,
            key_preview=key[:50],
            cache_size=len(self.cache)
        )
    
    def clear(self):
        """Limpa todo o cache"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do cache.
        
        Returns:
            Dict com estatísticas (tamanho atual, tamanho máximo, TTL)
        """
        return {
            "current_size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }

