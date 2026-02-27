from abc import ABC, abstractmethod
from typing import Any, Dict


class SignalInterface(ABC):
    """
    Contract for all analysis signals (Strategy Pattern).
    Every signal must inherit from this to ensure compatibility.
    """

    @property
    @abstractmethod
    def key(self) -> str:
        """Returns the string used as the key in the final JSON response (e.g. 'indecision_metrics')."""
        pass

    @abstractmethod
    async def run(self, text: str, analyzer: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run signal logic.

        Input: Raw string, BERTAnalyzer instance, and optional metadata (context).
        Output: A dictionary containing score (float), detected (bool), and metadata (dict).
        """
        pass