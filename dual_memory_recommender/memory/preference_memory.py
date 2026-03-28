from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dual_memory_recommender.schemas.core import PreferencePrototype, UserPreferenceProfile


class EvolvingMultimodalPreferenceMemory:
    """Prototype-oriented preference memory that merges reusable patterns across users."""

    def __init__(self, memory_path: str | Path) -> None:
        self.memory_path = Path(memory_path)
        self._prototypes: List[PreferencePrototype] = []
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.memory_path.exists():
            return
        payload = json.loads(self.memory_path.read_text(encoding="utf-8"))
        self._prototypes = [PreferencePrototype(**x) for x in payload.get("prototypes", [])]

    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        aa = set(x.lower() for x in a if x)
        bb = set(x.lower() for x in b if x)
        if not aa and not bb:
            return 0.0
        return len(aa & bb) / max(1, len(aa | bb))

    def _similarity(self, profile: UserPreferenceProfile, proto: PreferencePrototype) -> float:
        pat = proto.aggregated_preference_patterns
        s1 = self._jaccard(profile.preferred_attributes, pat.get("preferred_attributes", []))
        s2 = self._jaccard(profile.disliked_attributes, pat.get("disliked_attributes", []))
        s3 = self._jaccard(profile.style_cues, pat.get("style_cues", []))
        visual_match = 1.0 - min(abs(profile.visual_reliance - float(proto.applicable_conditions.get("visual_reliance", 0.5))), 1.0)
        return 0.35 * s1 + 0.2 * s2 + 0.25 * s3 + 0.2 * visual_match

    def retrieve(self, profile: UserPreferenceProfile, top_k: int = 3) -> List[Tuple[float, PreferencePrototype]]:
        self.load()
        scored = [(self._similarity(profile, p), p) for p in self._prototypes]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x for x in scored[:top_k] if x[0] > 0.15]

    def _merge_unique(self, old: List[str], new: List[str], max_n: int = 25) -> List[str]:
        out: List[str] = []
        seen = set()
        for x in old + new:
            t = str(x).strip()
            if not t:
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
            if len(out) >= max_n:
                break
        return out

    def _new_proto(self, profile: UserPreferenceProfile) -> PreferencePrototype:
        base_id = f"proto_{len(self._prototypes)+1:05d}"
        return PreferencePrototype(
            prototype_id=base_id,
            aggregated_preference_patterns={
                "preferred_attributes": list(profile.preferred_attributes),
                "disliked_attributes": list(profile.disliked_attributes),
                "style_cues": list(profile.style_cues),
                "functional_requirements": list(profile.functional_requirements),
            },
            multimodal_attribute_summary={
                "textual": list(profile.preferred_attributes),
                "visual": list(profile.style_cues),
            },
            positive_negative_patterns={
                "positive": list(profile.positive_preferences),
                "negative": list(profile.negative_preferences),
            },
            applicable_conditions={"visual_reliance": profile.visual_reliance},
            representative_users=[profile.user_id],
            support_count=1,
            confidence=float(profile.confidence_scores.get("overall", 0.5)),
            update_count=1,
        )

    def update(self, profile: UserPreferenceProfile) -> Dict[str, Any]:
        self.load()
        hits = self.retrieve(profile, top_k=1)
        if not hits or hits[0][0] < 0.4:
            self._prototypes.append(self._new_proto(profile))
            self._persist()
            return {"action": "create", "prototype_id": self._prototypes[-1].prototype_id}

        _, proto = hits[0]
        pat = proto.aggregated_preference_patterns
        pat["preferred_attributes"] = self._merge_unique(pat.get("preferred_attributes", []), profile.preferred_attributes)
        pat["disliked_attributes"] = self._merge_unique(pat.get("disliked_attributes", []), profile.disliked_attributes)
        pat["style_cues"] = self._merge_unique(pat.get("style_cues", []), profile.style_cues)
        pat["functional_requirements"] = self._merge_unique(pat.get("functional_requirements", []), profile.functional_requirements)
        proto.positive_negative_patterns["positive"] = self._merge_unique(proto.positive_negative_patterns.get("positive", []), profile.positive_preferences)
        proto.positive_negative_patterns["negative"] = self._merge_unique(proto.positive_negative_patterns.get("negative", []), profile.negative_preferences)
        proto.multimodal_attribute_summary["visual"] = self._merge_unique(proto.multimodal_attribute_summary.get("visual", []), profile.style_cues)
        proto.multimodal_attribute_summary["textual"] = self._merge_unique(proto.multimodal_attribute_summary.get("textual", []), profile.preferred_attributes)
        proto.representative_users = self._merge_unique(proto.representative_users, [profile.user_id], max_n=50)
        proto.support_count += 1
        proto.update_count += 1
        proto.confidence = min(0.99, (proto.confidence * 0.8) + 0.2 * float(profile.confidence_scores.get("overall", 0.5)))
        proto.applicable_conditions["visual_reliance"] = (float(proto.applicable_conditions.get("visual_reliance", 0.5)) + profile.visual_reliance) / 2.0
        self._persist()
        return {"action": "merge", "prototype_id": proto.prototype_id}

    def _persist(self) -> None:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"prototypes": [p.to_dict() for p in self._prototypes]}
        self.memory_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
