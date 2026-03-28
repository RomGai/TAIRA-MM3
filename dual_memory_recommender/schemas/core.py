from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class TaskSignature:
    user_id: str
    query_text: str
    category: str = ""
    query_length: int = 0
    query_granularity: str = "unknown"
    shopping_goal_type: str = "unknown"
    estimated_visual_reliance: float = 0.5
    history_richness: float = 0.0
    negative_signal_strength: float = 0.0
    session_id: str = ""
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalPolicy:
    tool_weights: Dict[str, float]
    tool_execution_order: List[str]
    per_tool_topk: Dict[str, int]
    query_rewrites: List[str] = field(default_factory=list)
    use_multimodal_retrieval: bool = True
    inject_collaborative_candidates: bool = True
    use_category_prefilter: bool = False
    category_prefilter_mode: str = "disabled"
    reflection_max_iters: int = 1
    reflection_stop_min_gain: float = 0.02
    confidence: float = 0.5
    source: str = "default"
    fallback_used: bool = False

    def normalized_weights(self) -> Dict[str, float]:
        keys = ["tool_b_text", "tool_c_multimodal", "tool_d_keyword"]
        vals = [max(0.0, float(self.tool_weights.get(k, 0.0))) for k in keys]
        s = sum(vals)
        if s <= 0:
            return {"tool_b_text": 0.5, "tool_c_multimodal": 0.3, "tool_d_keyword": 0.2}
        return {k: v / s for k, v in zip(keys, vals)}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HistoricalPseudoQuery:
    user_id: str
    history_item_id: str
    source_behavior_label: str
    generated_query_text: str
    query_style_tag: str
    source_current_query_text: str
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalExperience:
    task_signature: TaskSignature
    initial_policy: RetrievalPolicy
    final_policy: RetrievalPolicy
    retrieval_feedback_metrics: Dict[str, float]
    candidate_quality_indicators: Dict[str, float]
    historical_self_calibration: Dict[str, Any]
    worked: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    reflection_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["task_signature"] = self.task_signature.to_dict()
        payload["initial_policy"] = self.initial_policy.to_dict()
        payload["final_policy"] = self.final_policy.to_dict()
        return payload


@dataclass
class UserPreferenceProfile:
    user_id: str
    query_text: str
    positive_preferences: List[str]
    negative_preferences: List[str]
    preferred_attributes: List[str]
    disliked_attributes: List[str]
    style_cues: List[str]
    functional_requirements: List[str]
    visual_reliance: float
    confidence_scores: Dict[str, float]
    evidence_sources: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreferencePrototype:
    prototype_id: str
    aggregated_preference_patterns: Dict[str, List[str]]
    multimodal_attribute_summary: Dict[str, List[str]]
    positive_negative_patterns: Dict[str, List[str]]
    applicable_conditions: Dict[str, Any]
    representative_users: List[str] = field(default_factory=list)
    representative_items: List[str] = field(default_factory=list)
    support_count: int = 0
    confidence: float = 0.5
    update_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
