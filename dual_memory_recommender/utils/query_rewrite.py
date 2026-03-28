from __future__ import annotations

from typing import Any, Dict, List

from dual_memory_recommender.schemas.core import HistoricalPseudoQuery


def infer_query_style(query_text: str) -> str:
    q = (query_text or "").lower()
    style_tokens = ["style", "look", "vintage", "elegant", "casual", "color", "design"]
    func_tokens = ["waterproof", "durable", "comfortable", "lightweight", "battery", "performance"]
    s = sum(1 for t in style_tokens if t in q)
    f = sum(1 for t in func_tokens if t in q)
    if s > f:
        return "style_oriented"
    if f > s:
        return "function_oriented"
    return "mixed"


def build_historical_pseudo_queries(
    user_id: str,
    current_query_text: str,
    history_rows: List[Dict[str, Any]],
    max_queries: int = 8,
) -> List[HistoricalPseudoQuery]:
    style = infer_query_style(current_query_text)
    pseudo: List[HistoricalPseudoQuery] = []
    for row in history_rows[:max_queries]:
        profile = row.get("profile", {}) or {}
        title = str(profile.get("title", "")).strip()
        taxonomy = profile.get("taxonomy", {}) or {}
        item_type = taxonomy.get("item_type") or "item"
        text_tags = profile.get("text_tags", {}) or {}
        desc = str(text_tags.get("summary", "")).strip()
        if style == "style_oriented":
            generated = f"Find {item_type} with style cues similar to {title}. {desc[:120]}"
        elif style == "function_oriented":
            generated = f"Find {item_type} with functional properties matching {title}. {desc[:120]}"
        else:
            generated = f"Find {item_type} like {title} balancing style and functionality. {desc[:120]}"
        pseudo.append(
            HistoricalPseudoQuery(
                user_id=user_id,
                history_item_id=str(row.get("item_id", "")),
                source_behavior_label=str(row.get("behavior", "positive")),
                generated_query_text=" ".join(generated.split()),
                query_style_tag=style,
                source_current_query_text=current_query_text,
                confidence=0.6,
            )
        )
    return pseudo
