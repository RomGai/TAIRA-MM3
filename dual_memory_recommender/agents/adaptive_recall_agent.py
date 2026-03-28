from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, List, Tuple


from dual_memory_recommender.memory.retrieval_policy_memory import CollaborativeRetrievalPolicyMemory
from dual_memory_recommender.schemas.core import RetrievalExperience, RetrievalPolicy, TaskSignature
from dual_memory_recommender.utils.query_rewrite import build_historical_pseudo_queries
from dual_memory_recommender.utils.reflection import CalibrationFeedback, adjust_policy_from_feedback, summarize_feedback

logger = logging.getLogger(__name__)


class AdaptiveRecallAgent:
    def __init__(
        self,
        policy_memory: CollaborativeRetrievalPolicyMemory,
        *,
        use_category_prefilter: bool = False,
        category_prefilter_mode: str = "disabled",
        disable_retrieval_policy_memory: bool = False,
        disable_reflection: bool = False,
        fixed_tool_weights: Dict[str, float] | None = None,
        force_fixed_topk_fallback: bool = False,
        no_collaborative_augmentation: bool = False,
        fixed_recall_topk: int = 250,
    ) -> None:
        self.policy_memory = policy_memory
        self.use_category_prefilter = use_category_prefilter
        self.category_prefilter_mode = category_prefilter_mode
        self.disable_retrieval_policy_memory = disable_retrieval_policy_memory
        self.disable_reflection = disable_reflection
        self.fixed_tool_weights = fixed_tool_weights
        self.force_fixed_topk_fallback = force_fixed_topk_fallback
        self.no_collaborative_augmentation = no_collaborative_augmentation
        self.fixed_recall_topk = fixed_recall_topk

    def default_policy(self) -> RetrievalPolicy:
        weights = self.fixed_tool_weights or {"tool_b_text": 0.5, "tool_c_multimodal": 0.3, "tool_d_keyword": 0.2}
        return RetrievalPolicy(
            tool_weights=weights,
            tool_execution_order=["tool_b_text", "tool_c_multimodal", "tool_d_keyword"],
            per_tool_topk={"tool_b_text": self.fixed_recall_topk, "tool_c_multimodal": 25, "tool_d_keyword": self.fixed_recall_topk},
            query_rewrites=[],
            use_multimodal_retrieval=True,
            inject_collaborative_candidates=not self.no_collaborative_augmentation,
            use_category_prefilter=self.use_category_prefilter,
            category_prefilter_mode=self.category_prefilter_mode,
            reflection_max_iters=2,
            reflection_stop_min_gain=0.02,
            confidence=0.4,
            source="default",
            fallback_used=False,
        )

    def _category_prefilter(
        self,
        all_item_ids: List[str],
        meta_map: Dict[str, Dict[str, Any]],
        routed_category_paths: List[List[str]],
    ) -> List[str]:
        if not self.use_category_prefilter or self.category_prefilter_mode == "disabled" or not routed_category_paths:
            logger.info("[AdaptiveRecall] category prefilter disabled")
            return all_item_ids
        path_tokens = set(seg.lower() for p in routed_category_paths for seg in p)
        filtered = []
        for iid in all_item_ids:
            categories = meta_map.get(iid, {}).get("categories", [])
            text = " ".join(str(x).lower() for row in categories for x in (row if isinstance(row, list) else [row]))
            if any(tok in text for tok in path_tokens):
                filtered.append(iid)
        logger.info("[AdaptiveRecall] category prefilter enabled, kept=%s/%s", len(filtered), len(all_item_ids))
        return filtered or all_item_ids

    def _score_tools(
        self,
        candidate_ids: List[str],
        text_rank_ids: List[str],
        keyword_rank_ids: List[str],
        multimodal_rank_ids: List[str],
        policy: RetrievalPolicy,
    ) -> List[str]:
        weights = policy.normalized_weights()
        score: Dict[str, float] = {iid: 0.0 for iid in candidate_ids}

        for rank, iid in enumerate(text_rank_ids, start=1):
            if iid in score:
                score[iid] += weights["tool_b_text"] / rank
        for rank, iid in enumerate(multimodal_rank_ids, start=1):
            if iid in score and policy.use_multimodal_retrieval:
                score[iid] += weights["tool_c_multimodal"] / rank
        for rank, iid in enumerate(keyword_rank_ids, start=1):
            if iid in score:
                score[iid] += weights["tool_d_keyword"] / rank

        merged = sorted(score.items(), key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in merged if _ > 0]

    def _calibrate(
        self,
        policy: RetrievalPolicy,
        user_id: str,
        query_text: str,
        history_rows: List[Dict[str, Any]],
        retrieval_fn,
    ) -> Tuple[RetrievalPolicy, Dict[str, Any]]:
        if self.disable_reflection or not history_rows:
            return policy, {"reflection_used": False}

        pseudo_queries = build_historical_pseudo_queries(user_id=user_id, current_query_text=query_text, history_rows=history_rows)
        feedbacks: List[CalibrationFeedback] = []
        current = replace(policy)
        for _ in range(max(1, current.reflection_max_iters)):
            hits = 0
            total = 0
            for pq in pseudo_queries:
                recalled = retrieval_fn(pq.generated_query_text, current)
                total += 1
                if pq.history_item_id in recalled[: max(5, int(current.per_tool_topk.get("tool_b_text", 50) * 0.2))]:
                    hits += 1
            hit_rate = hits / max(1, total)
            fb = CalibrationFeedback(
                hit_rate_at_k=hit_rate,
                total=total,
                hits=hits,
                textual_signal=current.tool_weights.get("tool_b_text", 0.0),
                multimodal_signal=current.tool_weights.get("tool_c_multimodal", 0.0),
                category_drift=0.0 if current.use_category_prefilter else 0.6,
            )
            feedbacks.append(fb)
            prev = current.confidence
            current = adjust_policy_from_feedback(current, fb)
            if abs(current.confidence - prev) < current.reflection_stop_min_gain:
                break

        summary = summarize_feedback(feedbacks)
        return current, {
            "reflection_used": True,
            "pseudo_query_count": len(pseudo_queries),
            "summary": summary,
            "historical_pseudo_queries": [x.to_dict() for x in pseudo_queries],
        }

    def run(
        self,
        *,
        signature: TaskSignature,
        query_text: str,
        all_item_ids: List[str],
        meta_map: Dict[str, Dict[str, Any]],
        routed_category_paths: List[List[str]],
        text_rank_ids: List[str],
        keyword_rank_ids: List[str],
        multimodal_rank_ids: List[str],
        user_history_rows: List[Dict[str, Any]],
        collaborative_evidence: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        default_policy = self.default_policy()
        if self.force_fixed_topk_fallback:
            policy = default_policy
            policy.fallback_used = True
            memory_meta = {"memory_hit": False, "forced_fallback": True}
        elif self.disable_retrieval_policy_memory:
            policy = default_policy
            memory_meta = {"memory_hit": False, "disabled": True}
        else:
            policy, memory_meta = self.policy_memory.suggest_policy(signature, default_policy)

        initial_policy = replace(policy)

        filtered_ids = self._category_prefilter(all_item_ids, meta_map, routed_category_paths)

        def _retrieve_for_query(_query: str, p: RetrievalPolicy) -> List[str]:
            return self._score_tools(filtered_ids, text_rank_ids, keyword_rank_ids, multimodal_rank_ids, p)

        refined_policy, calibration = self._calibrate(
            policy=policy,
            user_id=signature.user_id,
            query_text=query_text,
            history_rows=user_history_rows,
            retrieval_fn=_retrieve_for_query,
        )

        if refined_policy.confidence < 0.2:
            refined_policy = default_policy
            refined_policy.fallback_used = True

        recalled_ids = _retrieve_for_query(query_text, refined_policy)
        if collaborative_evidence and refined_policy.inject_collaborative_candidates and not self.no_collaborative_augmentation:
            seeds = collaborative_evidence.get("collaborative_candidate_seeds", [])
            recalled_ids = list(dict.fromkeys([*seeds, *recalled_ids]))

        exp = RetrievalExperience(
            task_signature=signature,
            initial_policy=initial_policy,
            final_policy=refined_policy,
            retrieval_feedback_metrics={"recalled_count": float(len(recalled_ids)), "memory_hit": float(bool(memory_meta.get("memory_hit")))},
            candidate_quality_indicators={"unique_candidates": float(len(set(recalled_ids)))},
            historical_self_calibration=calibration,
            worked=["tool_b_text"] if refined_policy.tool_weights.get("tool_b_text", 0) >= 0.4 else [],
            failed=["tool_c_multimodal"] if refined_policy.tool_weights.get("tool_c_multimodal", 0) < 0.1 else [],
            reflection_summary=str(calibration.get("summary", {})),
        )
        if not self.disable_retrieval_policy_memory:
            self.policy_memory.update(exp)

        logger.info("[AdaptiveRecall] policy source=%s confidence=%.3f fallback=%s memory=%s", refined_policy.source, refined_policy.confidence, refined_policy.fallback_used, memory_meta)

        return {
            "candidate_ids": recalled_ids,
            "policy": refined_policy.to_dict(),
            "memory_meta": memory_meta,
            "calibration": calibration,
        }
