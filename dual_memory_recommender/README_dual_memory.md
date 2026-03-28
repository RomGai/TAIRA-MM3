# Dual-Memory Agentic Recommender (Cloth Pipeline Extension)

## Architecture mapping

This extension preserves the original 5-agent flow while replacing Agent3/Agent4 internals:

1. Agent1 Item profiling: still uses `GlobalItemDB` item profiles.
2. Agent2 User behavior profiling: still writes/reads `UserHistoryLogDB`.
3. Agent3 upgraded to `AdaptiveRecallAgent`:
   - Builds `TaskSignature`.
   - Retrieves `RetrievalPolicy` from `CollaborativeRetrievalPolicyMemory`.
   - Supports category prefilter as optional hard gate (`use_category_prefilter` + `category_prefilter_mode`).
   - Schedules only B/C/D tools with adaptive weights.
   - Runs user-history pseudo-query self-calibration.
   - Falls back to fixed strategy when confidence is low.
4. Agent4 upgraded to `CollaborativePreferenceAgent`:
   - Produces `UserPreferenceProfile`.
   - Retrieves and updates prototypes in `EvolvingMultimodalPreferenceMemory`.
   - Outputs collaborative evidence for Agent3.
5. Agent5 reranking: still uses `LLMItemReranker`, now with richer preference inputs.

## New entrypoint

- `dual_memory_recommender/run_cloth_unified_eval_pipeline_dual_memory.py`

## Example run command

```bash
python dual_memory_recommender/run_cloth_unified_eval_pipeline_dual_memory.py \
  --query-csv data/amazon_clothing/query_data1.csv \
  --filtered-meta-jsonl data/amazon_clothing/meta_Clothing_Shoes_and_Jewelry.filtered.jsonl \
  --output-dir processed/cloth_dual_memory_outputs \
  --retrieval-policy-memory-path processed/dual_memory/retrieval_policy_memory.jsonl \
  --preference-memory-path processed/dual_memory/preference_memory.json \
  --use-category-prefilter \
  --category-prefilter-mode hard_filter
```

## Ablation toggles

- `--disable-retrieval-policy-memory`
- `--disable-preference-memory`
- `--disable-reflection`
- `--fixed-tool-weights TEXT MM KW`
- `--no-collaborative-augmentation`
- `--use-category-prefilter`
- `--force-fixed-topk-fallback`

## TODOs / approximation notes

- Current multimodal ranking branch uses text-rank proxy unless `--enable-agent3-qwen3vl-embedding` and a multimodal index is added.
- Preference extraction is currently heuristic and can be upgraded with an LLM parser.
- Policy-memory similarity is handcrafted for lightweight local experimentation.
