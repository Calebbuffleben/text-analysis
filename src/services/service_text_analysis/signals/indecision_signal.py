from typing import Any, Dict

from ....signals.indecision import compute_indecision_metrics_safe
from .signal_interface import SignalInterface

class IndecisionSignal(SignalInterface):
    """
    Expert for detecting hesitation and conditional language.
    Encapsulates: Keyword Extraction, Conditional Detection, Scoring.
    Stateless; any state (e.g. sales_category) comes via context.
    """

    @property
    def key(self) -> str:
        return "indecision_metrics"

    async def run(self, text: str, analyzer: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        # Keyword Extraction (spec §4)
        keywords = analyzer.extract_keywords(text, top_n=10)
        # Conditional Detection (spec §4)
        conditional_keywords_detected = []
        if context.get("sbert_enabled") and hasattr(analyzer, "detect_conditional_keywords"):
            try:
                conditional_keywords_detected = analyzer.detect_conditional_keywords(text, keywords)
            except Exception:
                conditional_keywords_detected = []
        # Scoring; sales_category* from context (orchestrator precomputes)
        semantic = context.get("semantic_result") or {}
        metrics = compute_indecision_metrics_safe(
            analyzer=analyzer,
            sbert_enabled=context.get("sbert_enabled", False),
            sales_category=semantic.get("sales_category"),
            sales_category_confidence=semantic.get("sales_category_confidence"),
            sales_category_intensity=semantic.get("sales_category_intensity"),
            sales_category_ambiguity=semantic.get("sales_category_ambiguity"),
            conditional_keywords_detected=conditional_keywords_detected,
            meeting_id=context.get("meeting_id", ""),
        )
        score = float(metrics.get("indecision_score", 0.0) or 0.0)
        detected = score >= 0.2
        return {"score": score, "detected": detected, "metadata": metrics}