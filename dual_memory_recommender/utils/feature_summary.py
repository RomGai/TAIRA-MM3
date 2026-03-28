from __future__ import annotations

from typing import Any, Dict, List

from dual_memory_recommender.schemas.core import TaskSignature
from dual_memory_recommender.utils.query_rewrite import infer_query_style


def build_task_signature(
    user_id: str,
    query_text: str,
    routed_category: str,
    history_rows: List[Dict[str, Any]],
) -> TaskSignature:
    neg = [x for x in history_rows if str(x.get("behavior", "")).lower() in {"negative", "dislike", "skip"}]
    history_richness = min(1.0, len(history_rows) / 50.0)
    negative_strength = min(1.0, len(neg) / max(1, len(history_rows))) if history_rows else 0.0
    style = infer_query_style(query_text)
    goal = "mixed"
    if style == "style_oriented":
        goal = "style-seeking"
    elif style == "function_oriented":
        goal = "function-seeking"
    visual = 0.75 if style == "style_oriented" else 0.35 if style == "function_oriented" else 0.55
    granularity = "specific" if len(query_text.split()) >= 8 else "broad"

    return TaskSignature(
        user_id=user_id,
        query_text=query_text,
        category=routed_category,
        query_length=len(query_text.split()),
        query_granularity=granularity,
        shopping_goal_type=goal,
        estimated_visual_reliance=visual,
        history_richness=history_richness,
        negative_signal_strength=negative_strength,
    )
