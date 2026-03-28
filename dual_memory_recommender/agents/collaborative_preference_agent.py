from __future__ import annotations

from typing import Any, Dict, List

from dual_memory_recommender.memory.preference_memory import EvolvingMultimodalPreferenceMemory
from dual_memory_recommender.schemas.core import UserPreferenceProfile


def _extract_attributes(history_rows: List[Dict[str, Any]], behavior: str) -> List[str]:
    attrs: List[str] = []
    for row in history_rows:
        if str(row.get("behavior", "positive")).lower() != behavior:
            continue
        profile = row.get("profile", {}) or {}
        taxonomy = profile.get("taxonomy", {}) or {}
        text_tags = profile.get("text_tags", {}) or {}
        title = str(profile.get("title", "")).strip()
        item_type = str(taxonomy.get("item_type", "")).strip()
        summary = str(text_tags.get("summary", "")).strip()
        for token in [item_type, title, summary[:80]]:
            t = " ".join(token.split())
            if t:
                attrs.append(t)
    return attrs


class CollaborativePreferenceAgent:
    def __init__(self, preference_memory: EvolvingMultimodalPreferenceMemory, disable_preference_memory: bool = False) -> None:
        self.preference_memory = preference_memory
        self.disable_preference_memory = disable_preference_memory

    def infer(
        self,
        user_id: str,
        query_text: str,
        history_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        pos = _extract_attributes(history_rows, "positive")
        neg = _extract_attributes(history_rows, "negative")

        profile = UserPreferenceProfile(
            user_id=user_id,
            query_text=query_text,
            positive_preferences=pos[:12],
            negative_preferences=neg[:12],
            preferred_attributes=pos[:10],
            disliked_attributes=neg[:8],
            style_cues=[x for x in pos if any(k in x.lower() for k in ["style", "design", "color", "vintage"])][:8],
            functional_requirements=[x for x in pos if any(k in x.lower() for k in ["durable", "comfort", "water", "fit", "performance"])][:8],
            visual_reliance=0.7 if len(pos) > 0 and len([x for x in pos if "color" in x.lower() or "style" in x.lower()]) > 0 else 0.45,
            confidence_scores={"overall": 0.7 if history_rows else 0.3, "history_support": min(1.0, len(history_rows) / 20.0)},
            evidence_sources={"history_count": len(history_rows), "query": query_text},
        )

        memory_hits: List[Dict[str, Any]] = []
        collaborative_candidates: List[str] = []
        if not self.disable_preference_memory:
            retrieved = self.preference_memory.retrieve(profile, top_k=3)
            for score, proto in retrieved:
                memory_hits.append({"prototype_id": proto.prototype_id, "score": score, "support_count": proto.support_count})
                collaborative_candidates.extend(proto.representative_items[:10])
                profile.preferred_attributes = list(dict.fromkeys(profile.preferred_attributes + proto.aggregated_preference_patterns.get("preferred_attributes", [])[:4]))
                profile.disliked_attributes = list(dict.fromkeys(profile.disliked_attributes + proto.aggregated_preference_patterns.get("disliked_attributes", [])[:2]))

            update_info = self.preference_memory.update(profile)
        else:
            update_info = {"action": "disabled"}

        evidence = {
            "refined_attributes": profile.preferred_attributes[:15],
            "collaborative_candidate_seeds": collaborative_candidates[:30],
            "visual_reliance_hint": profile.visual_reliance,
            "memory_hits": memory_hits,
            "memory_update": update_info,
        }

        return {
            "user_preference_profile": profile.to_dict(),
            "collaborative_evidence": evidence,
        }
