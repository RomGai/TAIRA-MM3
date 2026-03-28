from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from dual_memory_recommender.schemas.core import RetrievalPolicy


@dataclass
class CalibrationFeedback:
    hit_rate_at_k: float
    total: int
    hits: int
    textual_signal: float
    multimodal_signal: float
    category_drift: float


def adjust_policy_from_feedback(policy: RetrievalPolicy, feedback: CalibrationFeedback, min_confidence: float = 0.2) -> RetrievalPolicy:
    weights = policy.normalized_weights()
    if feedback.total <= 0:
        policy.fallback_used = True
        policy.confidence = 0.0
        return policy

    if feedback.hit_rate_at_k < 0.2:
        weights["tool_d_keyword"] = min(0.5, weights["tool_d_keyword"] + 0.1)
        weights["tool_b_text"] = min(0.7, weights["tool_b_text"] + 0.1)
        weights["tool_c_multimodal"] = max(0.1, weights["tool_c_multimodal"] - 0.2)
    else:
        if feedback.multimodal_signal > feedback.textual_signal:
            weights["tool_c_multimodal"] = min(0.7, weights["tool_c_multimodal"] + 0.1)
        else:
            weights["tool_b_text"] = min(0.7, weights["tool_b_text"] + 0.1)

    if feedback.category_drift > 0.5:
        policy.use_category_prefilter = True
        policy.category_prefilter_mode = "hard_filter"

    total = sum(weights.values())
    policy.tool_weights = {k: v / total for k, v in weights.items()}
    policy.confidence = max(min_confidence, feedback.hit_rate_at_k)
    return policy


def summarize_feedback(feedbacks: List[CalibrationFeedback]) -> Dict[str, float]:
    if not feedbacks:
        return {"hit_rate_at_k": 0.0, "textual_signal": 0.0, "multimodal_signal": 0.0, "category_drift": 0.0}
    n = len(feedbacks)
    return {
        "hit_rate_at_k": sum(f.hit_rate_at_k for f in feedbacks) / n,
        "textual_signal": sum(f.textual_signal for f in feedbacks) / n,
        "multimodal_signal": sum(f.multimodal_signal for f in feedbacks) / n,
        "category_drift": sum(f.category_drift for f in feedbacks) / n,
    }
