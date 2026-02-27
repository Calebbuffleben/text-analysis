from typing import Any, Dict

from ....signals.reformulation import (
    apply_solution_reformulation_signal_flag,
    compute_reformulation_marker_score,
    detect_reformulation_markers,
)
from .signal_interface import SignalInterface


class ReformulationSignal(SignalInterface):
    """Expert for reformulation/teach-back markers. Stateless; uses text only for detection."""

    @property
    def key(self) -> str:
        return "reformulation"

    async def run(self, text: str, analyzer: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        markers = detect_reformulation_markers(text)
        score = compute_reformulation_marker_score(markers)
        semantic = context.get("semantic_result") or {}
        flags = semantic.get("sales_category_flags")
        if isinstance(flags, dict):
            apply_solution_reformulation_signal_flag(flags, score)
        metadata = {
            "reformulation_markers_detected": markers,
            "reformulation_marker_score": score,
        }
        return {"score": score, "detected": score > 0.0, "metadata": metadata}
