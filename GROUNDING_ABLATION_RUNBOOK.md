# Grounding Ablation Runbook

This runbook covers the data-only PrimeKG grounding ablation flow used for MedReason seed building.

## Modes

- `baseline_current`
  - Uses the current `AgenticRetriever.retrieve()` path unchanged.
- `deterministic_v2`
  - Uses data-only node catalog grounding with exact, alias, suffix, and fuzzy lexical matching.
- `llm_assisted_v1`
  - Uses deterministic candidates first, then an LLM judge reranks or skips from supplied candidates only.

## Seed Building

Use `prepare_medreason_seed_jsonl(...)` with `grounding_mode`:

```python
prepare_medreason_seed_jsonl(
    source="data/medreason",
    output_jsonl=Path("artifacts/trm_seed.jsonl"),
    retriever=retriever,
    grounding_mode="deterministic_v2",
    edge_mapper_backend="heuristic",
)
```

Seed row metadata now includes:

- `grounding_mode`
- `grounding_backend`
- `entity_grounding`
- `unresolved_entities`
- `seed_node_ids`
- `candidate_stats`
- `llm_grounding_used`
- `llm_grounding_fallback_reason`

## Analysis Commands

Single-sample trace:

```powershell
python tools\grounding_ablation.py trace `
  --source data/medreason `
  --split train `
  --sample-index 0 `
  --grounding-mode deterministic_v2
```

Batch audit:

```powershell
python tools\grounding_ablation.py audit `
  --source data/medreason `
  --split train `
  --limit 100 `
  --grounding-mode deterministic_v2 `
  --output-json artifacts\grounding_audit_det_v2.json
```

3-way ablation:

```powershell
python tools\grounding_ablation.py ablation `
  --source data/medreason `
  --split train `
  --limit 100 `
  --output-json artifacts\grounding_ablation.json
```

## Notes

- This flow is data-only and does not change inference runtime.
- The analysis CLI disables PubMed by default to stay lightweight.
- `llm_assisted_v1` falls back to deterministic grounding when no real LLM backend is configured.
