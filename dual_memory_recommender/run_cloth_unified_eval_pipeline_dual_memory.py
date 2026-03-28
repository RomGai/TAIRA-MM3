from __future__ import annotations

import argparse
import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from dual_memory_recommender.agents.adaptive_recall_agent import AdaptiveRecallAgent
from dual_memory_recommender.agents.collaborative_preference_agent import CollaborativePreferenceAgent
from dual_memory_recommender.memory.preference_memory import EvolvingMultimodalPreferenceMemory
from dual_memory_recommender.memory.retrieval_policy_memory import CollaborativeRetrievalPolicyMemory
from dual_memory_recommender.utils.feature_summary import build_task_signature
from reranker import LLMItemReranker

try:
    from item_profiler_agents import GlobalItemDB, UserHistoryLogDB
except ModuleNotFoundError:
    from new_pipe.item_profiler_agents import GlobalItemDB, UserHistoryLogDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dual_memory_pipeline")

EN_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "for", "with", "of", "in", "on", "at", "from", "by", "is", "are", "be",
}


def _parse_meta_line(line: str) -> dict:
    t = line.strip()
    if not t:
        return {}
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        return ast.literal_eval(t)


def load_filtered_meta(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = _parse_meta_line(line)
            asin = str(rec.get("asin", "")).strip()
            if asin:
                out[asin] = rec
    return out


def _meta_category_paths(meta: Dict[str, Any]) -> List[List[str]]:
    categories = meta.get("categories", [])
    out: List[List[str]] = []
    if isinstance(categories, list):
        for cat_path in categories:
            if isinstance(cat_path, list):
                segs = [str(x).strip() for x in cat_path if str(x).strip()]
                if segs:
                    out.append(segs)
    return out


def _meta_category_text(meta: Dict[str, Any]) -> str:
    return " | ".join(" > ".join(x) for x in _meta_category_paths(meta))


def _item_sentence(meta: Dict[str, Any]) -> str:
    return f"categories: {_meta_category_text(meta)}; title: {meta.get('title','')}; description: {meta.get('description','')}"


def _query_sentence(query: str, selected_categories: List[List[str]] | None = None) -> str:
    cats = " | ".join(" > ".join(seg for seg in c if seg) for c in (selected_categories or []))
    return f"categories: {cats}; user_need: {query}".strip()


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _extract_query_keywords(query: str, max_keywords: int) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", query.lower())
    uniq: List[str] = []
    seen = set()
    for t in tokens:
        if t in EN_STOPWORDS or len(t) <= 1:
            continue
        if t not in seen:
            seen.add(t)
            uniq.append(t)
        if len(uniq) >= max_keywords:
            break
    return uniq


def _rank_by_keyword(item_ids: List[str], title_lower_map: Dict[str, str], keywords: List[str], topk: int) -> List[str]:
    scored = []
    for iid in item_ids:
        title = title_lower_map.get(iid, "")
        score = sum(1 for kw in keywords if kw in title)
        if score > 0:
            scored.append((score, iid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [iid for _, iid in scored[:topk]]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    query_df = pd.read_csv(args.query_csv, dtype={"id": str, "user_id": str})
    if args.max_users > 0:
        query_df = query_df.head(args.max_users)

    meta_map = load_filtered_meta(Path(args.filtered_meta_jsonl))
    all_item_ids = sorted(meta_map.keys())
    item_id_to_index = {iid: idx for idx, iid in enumerate(all_item_ids)}
    title_lower_map = {iid: str(meta_map[iid].get("title", "") or "").lower() for iid in all_item_ids}

    emb_model = SentenceTransformer(args.embedding_model)
    item_sentences = [_item_sentence(meta_map[i]) for i in all_item_ids]
    item_emb = emb_model.encode(item_sentences, batch_size=args.embed_batch_size, convert_to_numpy=True, show_progress_bar=False)
    item_emb = _l2_normalize(item_emb.astype(np.float32, copy=False))

    global_db = GlobalItemDB(args.global_db)
    history_db = UserHistoryLogDB(args.history_db)

    policy_memory = CollaborativeRetrievalPolicyMemory(args.retrieval_policy_memory_path)
    pref_memory = EvolvingMultimodalPreferenceMemory(args.preference_memory_path)
    pref_agent = CollaborativePreferenceAgent(pref_memory, disable_preference_memory=args.disable_preference_memory)
    recall_agent = AdaptiveRecallAgent(
        policy_memory,
        use_category_prefilter=args.use_category_prefilter,
        category_prefilter_mode=args.category_prefilter_mode,
        disable_retrieval_policy_memory=args.disable_retrieval_policy_memory,
        disable_reflection=args.disable_reflection,
        fixed_tool_weights={"tool_b_text": args.fixed_tool_weights[0], "tool_c_multimodal": args.fixed_tool_weights[1], "tool_d_keyword": args.fixed_tool_weights[2]} if args.fixed_tool_weights else None,
        force_fixed_topk_fallback=args.force_fixed_topk_fallback,
        no_collaborative_augmentation=args.no_collaborative_augmentation,
        fixed_recall_topk=args.fixed_recall_topk,
    )
    reranker = LLMItemReranker(model_name=args.text_model)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    results = []
    for _, row in query_df.iterrows():
        user_id = str(row["user_id"])
        target_id = str(row["id"])
        query = str(row.get("new_query") or row.get("query") or "").strip()
        if not query:
            continue

        history_ids = [x for x in str(row.get("remaining_interaction_string", "")).split("|") if x]
        history_rows = []
        for iid in history_ids:
            profile = global_db.get_profile(iid)
            if profile is None:
                meta = meta_map.get(iid, {})
                profile = {"title": str(meta.get("title", "")), "taxonomy": {"item_type": ""}, "text_tags": {"summary": str(meta.get("description", ""))}, "visual_tags": {}}
                global_db.upsert(iid, profile)
            history_db.insert(user_id=user_id, item_id=iid, behavior="positive", timestamp=None, profile=profile)
            history_rows.append({"user_id": user_id, "item_id": iid, "behavior": "positive", "profile": profile})

        pref_out = pref_agent.infer(user_id=user_id, query_text=query, history_rows=history_rows)
        collaborative = pref_out["collaborative_evidence"]

        q_sentence = _query_sentence(query)
        q_emb = emb_model.encode([q_sentence], convert_to_numpy=True, show_progress_bar=False).astype(np.float32, copy=False)
        q_emb = _l2_normalize(q_emb)
        sim = np.matmul(item_emb, q_emb[0])
        text_rank = [all_item_ids[int(i)] for i in np.argsort(-sim)[: args.embedding_recall_topk]]

        keywords = _extract_query_keywords(query, args.max_query_keywords)
        keyword_rank = _rank_by_keyword(all_item_ids, title_lower_map, keywords, args.keyword_recall_topk)
        multimodal_rank = text_rank[: args.agent3_qwen3vl_topk] if args.enable_agent3_qwen3vl_embedding else []

        signature = build_task_signature(
            user_id=user_id,
            query_text=query,
            routed_category="",
            history_rows=history_rows,
        )

        recall_out = recall_agent.run(
            signature=signature,
            query_text=query,
            all_item_ids=all_item_ids,
            meta_map=meta_map,
            routed_category_paths=[],
            text_rank_ids=text_rank,
            keyword_rank_ids=keyword_rank,
            multimodal_rank_ids=multimodal_rank,
            user_history_rows=history_rows,
            collaborative_evidence=collaborative,
        )

        candidate_ids = recall_out["candidate_ids"][: args.top_n * 5]
        candidate_items = []
        for iid in candidate_ids:
            profile = global_db.get_profile(iid)
            if profile is None:
                meta = meta_map.get(iid, {})
                profile = {"title": str(meta.get("title", "")), "taxonomy": {"item_type": ""}, "text_tags": {"summary": str(meta.get("description", ""))}, "visual_tags": {}}
                global_db.upsert(iid, profile)
            candidate_items.append({"item_id": iid, "profile": profile})

        if not candidate_items:
            ranked_items = []
        else:
            ranked_items = reranker.rerank_items(query=query, preference_constraints={
                "Must_Have": pref_out["user_preference_profile"].get("functional_requirements", []),
                "Nice_to_Have": pref_out["user_preference_profile"].get("preferred_attributes", []),
                "Must_Avoid": pref_out["user_preference_profile"].get("disliked_attributes", []),
                "Predicted_Next_Items": [],
            }, candidate_items=candidate_items, top_n=args.top_n)

        payload = {
            "user_id": user_id,
            "query": query,
            "groundtruth_target_item_id": target_id,
            "user_preference_profile": pref_out["user_preference_profile"],
            "collaborative_evidence": collaborative,
            "adaptive_policy": recall_out["policy"],
            "memory_meta": recall_out["memory_meta"],
            "calibration": recall_out["calibration"],
            "ranked_items": ranked_items,
        }
        out_path = Path(args.output_dir) / f"user_{user_id}_dual_memory_output.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        hit = 1 if target_id in [x.get("item_id") for x in ranked_items[: args.top_n]] else 0
        results.append({"user_id": user_id, "hit": hit, "target_id": target_id})

    summary = {"rows": len(results), "recall@top_n": float(np.mean([r["hit"] for r in results])) if results else 0.0}
    Path(args.output_dir, "dual_memory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("summary=%s", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dual-memory cloth eval pipeline")
    parser.add_argument("--query-csv", default="data/amazon_clothing/query_data1.csv")
    parser.add_argument("--filtered-meta-jsonl", default="data/amazon_clothing/meta_Clothing_Shoes_and_Jewelry.filtered.jsonl")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--fixed-recall-topk", type=int, default=250)
    parser.add_argument("--keyword-recall-topk", type=int, default=250)
    parser.add_argument("--embedding-recall-topk", type=int, default=250)
    parser.add_argument("--agent3-qwen3vl-topk", type=int, default=25)
    parser.add_argument("--enable-agent3-qwen3vl-embedding", action="store_true")
    parser.add_argument("--max-query-keywords", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--max-users", type=int, default=0)

    parser.add_argument("--output-dir", default="processed/cloth_dual_memory_outputs")
    parser.add_argument("--global-db", default="processed/cloth_global_item_features.db")
    parser.add_argument("--history-db", default="processed/cloth_user_history.db")
    parser.add_argument("--text-model", default="Qwen/Qwen3-8B")

    parser.add_argument("--retrieval-policy-memory-path", default="processed/dual_memory/retrieval_policy_memory.jsonl")
    parser.add_argument("--preference-memory-path", default="processed/dual_memory/preference_memory.json")

    parser.add_argument("--disable-retrieval-policy-memory", action="store_true")
    parser.add_argument("--disable-preference-memory", action="store_true")
    parser.add_argument("--disable-reflection", action="store_true")
    parser.add_argument("--no-collaborative-augmentation", action="store_true")
    parser.add_argument("--force-fixed-topk-fallback", action="store_true")

    parser.add_argument("--use-category-prefilter", action="store_true")
    parser.add_argument("--category-prefilter-mode", default="hard_filter", choices=["hard_filter", "disabled"])
    parser.add_argument("--fixed-tool-weights", type=float, nargs=3, metavar=("TEXT", "MM", "KW"), default=None)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
