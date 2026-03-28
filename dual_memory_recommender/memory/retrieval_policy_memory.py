from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dual_memory_recommender.schemas.core import RetrievalExperience, RetrievalPolicy, TaskSignature


@dataclass
class PolicyMemoryHit:
    score: float
    experience: RetrievalExperience


class CollaborativeRetrievalPolicyMemory:
    """Lightweight JSONL-backed memory for task signature -> retrieval policy reuse."""

    def __init__(self, memory_path: str | Path) -> None:
        self.memory_path = Path(memory_path)
        self._experiences: List[RetrievalExperience] = []
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.memory_path.exists():
            return
        for line in self.memory_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            self._experiences.append(self._from_dict(rec))

    def _from_dict(self, rec: Dict[str, Any]) -> RetrievalExperience:
        ts = TaskSignature(**rec["task_signature"])
        init_policy = RetrievalPolicy(**rec["initial_policy"])
        final_policy = RetrievalPolicy(**rec["final_policy"])
        return RetrievalExperience(
            task_signature=ts,
            initial_policy=init_policy,
            final_policy=final_policy,
            retrieval_feedback_metrics=rec.get("retrieval_feedback_metrics", {}),
            candidate_quality_indicators=rec.get("candidate_quality_indicators", {}),
            historical_self_calibration=rec.get("historical_self_calibration", {}),
            worked=rec.get("worked", []),
            failed=rec.get("failed", []),
            reflection_summary=rec.get("reflection_summary", ""),
        )

    @staticmethod
    def _task_similarity(a: TaskSignature, b: TaskSignature) -> float:
        score = 0.0
        score += 0.35 if a.category and a.category == b.category else 0.0
        score += 0.2 * (1.0 - min(abs(a.query_length - b.query_length), 20) / 20.0)
        score += 0.15 if a.shopping_goal_type == b.shopping_goal_type else 0.0
        score += 0.15 * (1.0 - min(abs(a.estimated_visual_reliance - b.estimated_visual_reliance), 1.0))
        score += 0.15 * (1.0 - min(abs(a.history_richness - b.history_richness), 1.0))
        return max(0.0, min(1.0, score))

    def nearest(self, signature: TaskSignature, min_score: float = 0.45) -> Optional[PolicyMemoryHit]:
        self.load()
        best: Optional[PolicyMemoryHit] = None
        for exp in self._experiences:
            s = self._task_similarity(signature, exp.task_signature)
            if s < min_score:
                continue
            if best is None or s > best.score:
                best = PolicyMemoryHit(score=s, experience=exp)
        return best

    def suggest_policy(self, signature: TaskSignature, default_policy: RetrievalPolicy) -> Tuple[RetrievalPolicy, Dict[str, Any]]:
        hit = self.nearest(signature)
        if hit is None:
            return default_policy, {"memory_hit": False, "score": 0.0}
        policy = hit.experience.final_policy
        policy.source = "policy_memory"
        policy.confidence = min(0.95, max(policy.confidence, hit.score))
        return policy, {"memory_hit": True, "score": hit.score}

    def update(self, experience: RetrievalExperience) -> None:
        self.load()
        self._experiences.append(experience)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        with self.memory_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(experience.to_dict(), ensure_ascii=False) + "\n")
