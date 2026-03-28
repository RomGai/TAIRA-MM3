from pathlib import Path

from dual_memory_recommender.agents.adaptive_recall_agent import AdaptiveRecallAgent
from dual_memory_recommender.memory.preference_memory import EvolvingMultimodalPreferenceMemory
from dual_memory_recommender.memory.retrieval_policy_memory import CollaborativeRetrievalPolicyMemory
from dual_memory_recommender.schemas.core import TaskSignature, UserPreferenceProfile


def test_preference_memory_create_and_retrieve(tmp_path: Path):
    mem = EvolvingMultimodalPreferenceMemory(tmp_path / "pref.json")
    profile = UserPreferenceProfile(
        user_id="u1",
        query_text="casual vintage jacket",
        positive_preferences=["vintage", "casual"],
        negative_preferences=["formal"],
        preferred_attributes=["vintage", "casual"],
        disliked_attributes=["formal"],
        style_cues=["vintage color"],
        functional_requirements=["comfortable fit"],
        visual_reliance=0.7,
        confidence_scores={"overall": 0.8},
        evidence_sources={},
    )
    mem.update(profile)
    hits = mem.retrieve(profile, top_k=1)
    assert hits
    assert hits[0][1].support_count >= 1


def test_adaptive_agent_forced_fallback(tmp_path: Path):
    policy_mem = CollaborativeRetrievalPolicyMemory(tmp_path / "policy.jsonl")
    agent = AdaptiveRecallAgent(policy_mem, force_fixed_topk_fallback=True, fixed_recall_topk=10)
    sig = TaskSignature(user_id="u1", query_text="waterproof boots", query_length=2)
    out = agent.run(
        signature=sig,
        query_text="waterproof boots",
        all_item_ids=["i1", "i2", "i3"],
        meta_map={"i1": {}, "i2": {}, "i3": {}},
        routed_category_paths=[],
        text_rank_ids=["i2", "i1"],
        keyword_rank_ids=["i3"],
        multimodal_rank_ids=[],
        user_history_rows=[],
        collaborative_evidence=None,
    )
    assert out["policy"]["fallback_used"] is True
    assert len(out["candidate_ids"]) > 0
