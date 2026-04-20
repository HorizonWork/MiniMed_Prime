#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layers.layer1_retrieval import AgenticRetriever  # noqa: E402
from src.retrieval import infer_question_type  # noqa: E402
from src.training.medreason_adapter import DEFAULT_MEDREASON_SOURCE, load_medreason_records  # noqa: E402
from src.training.trm_dataset_builder import parse_reasoning_chain  # noqa: E402
from src.utils.kaggle_env import KaggleEnv, T4Hardening  # noqa: E402


@dataclass(slots=True)
class BenchmarkRow:
    index: int
    question_type: str
    question: str
    edges_without_filter: int
    edges_with_filter: int
    non_empty_without_filter: bool
    non_empty_with_filter: bool
    fallback_stage_used: str
    edges_before_filtering: int
    edges_after_filtering: int
    node_count_after_filtering: int
    gold_edge_count: int
    gold_edge_retention_without_filter: float
    gold_edge_retention_with_filter: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PrimeKG retrieval with relation filter disabled vs enabled.")
    parser.add_argument("--source", default=DEFAULT_MEDREASON_SOURCE, help="MedReason source path or HF name.")
    parser.add_argument("--split", default="train", help="Split name. Use 'none' for local unsplit jsonl.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum number of rows to benchmark.")
    parser.add_argument("--primekg-path", type=Path, default=Path("data/kg/primekg"))
    parser.add_argument("--pubmed-cache-path", type=Path, default=Path("data/pubmed_cache.jsonl"))
    parser.add_argument("--kaggle", action="store_true", help="Enable Kaggle path and T4 memory handling.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    return parser.parse_args()


def _iter_benchmark_rows(records: Sequence[Mapping[str, Any]], retriever: AgenticRetriever) -> Iterable[BenchmarkRow]:
    for index, record in enumerate(records):
        question = _question_text(record)
        if not question:
            continue
        question_type = infer_question_type(record=record, question=question, fallback_label="other")
        linked_entities = retriever.linker.link(question)
        resolved_entities = retriever.kg_extractor.resolve_seed_entities(linked_entities)
        edges_without_filter = retriever.kg_extractor.extract_2hop(
            seed_entities=resolved_entities,
            top_k=retriever.config.primekg_top_k,
            relation_filter=retriever.config.relation_filter,
            question_type=question_type,
            use_relation_filter=False,
            relation_filter_overrides=retriever.config.relation_filter_overrides,
        )
        edges_with_filter = retriever.kg_extractor.extract_2hop(
            seed_entities=resolved_entities,
            top_k=retriever.config.primekg_top_k,
            relation_filter=retriever.config.relation_filter,
            question_type=question_type,
            use_relation_filter=True,
            relation_filter_overrides=retriever.config.relation_filter_overrides,
        )
        relation_stats = dict(retriever.kg_extractor.last_relation_filter_stats)
        gold_edge_ids = set(parse_reasoning_chain(record))
        edge_ids_without_filter = {edge.edge_id for edge in edges_without_filter}
        edge_ids_with_filter = {edge.edge_id for edge in edges_with_filter}
        gold_edge_count = len(gold_edge_ids)
        if gold_edge_count > 0:
            retention_without_filter = len(gold_edge_ids & edge_ids_without_filter) / gold_edge_count
            retention_with_filter = len(gold_edge_ids & edge_ids_with_filter) / gold_edge_count
        else:
            retention_without_filter = 0.0
            retention_with_filter = 0.0

        yield BenchmarkRow(
            index=index,
            question_type=question_type,
            question=question,
            edges_without_filter=len(edges_without_filter),
            edges_with_filter=len(edges_with_filter),
            non_empty_without_filter=bool(edges_without_filter),
            non_empty_with_filter=bool(edges_with_filter),
            fallback_stage_used=str(relation_stats.get("fallback_stage_used") or relation_stats.get("fallback_stage") or "unknown"),
            edges_before_filtering=int(relation_stats.get("edges_before") or 0),
            edges_after_filtering=int(relation_stats.get("edges_after") or 0),
            node_count_after_filtering=int(relation_stats.get("node_count_after") or 0),
            gold_edge_count=gold_edge_count,
            gold_edge_retention_without_filter=retention_without_filter,
            gold_edge_retention_with_filter=retention_with_filter,
        )


def _load_records_or_fallback(source: str, split: str | None, limit: int) -> list[Mapping[str, Any]]:
    try:
        records = load_medreason_records(source, split=split, limit=limit)
    except Exception:
        records = []
    if records:
        return records[:limit]

    fallback_questions = [
        "Is ibuprofen contraindicated with warfarin in atrial fibrillation?",
        "What dose adjustment is needed for metformin in kidney disease?",
        "Most likely diagnosis for fever and lobar infiltrate?",
        "What causes diabetic ketoacidosis in type 1 diabetes?",
        "Can clopidogrel interact with omeprazole?",
        "Appropriate dosage of amoxicillin for pediatric otitis media?",
        "Etiology of peptic ulcer disease with Helicobacter pylori?",
        "Differential diagnosis for chest pain radiating to left arm?",
        "Is rivaroxaban contraindicated in severe renal impairment?",
        "Does atorvastatin interact with clarithromycin?",
    ]
    return [{"question": question} for question in fallback_questions[:limit]]


def _question_text(record: Mapping[str, Any]) -> str:
    for key in ("question", "question_text", "query", "prompt", "problem", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def main() -> int:
    args = _parse_args()
    if args.kaggle or KaggleEnv.is_kaggle():
        T4Hardening.setup_memory()

    split = None if str(args.split).lower() in {"", "none", "null", "false"} else args.split
    retriever = AgenticRetriever(
        primekg_path=KaggleEnv.path(args.primekg_path),
        pubmed_cache_path=KaggleEnv.ensure_writeable(KaggleEnv.path(args.pubmed_cache_path)),
    )
    # Keep benchmark lightweight and deterministic: use rule-based linker path only.
    retriever.linker._uses_umls = False  # type: ignore[attr-defined]
    retriever.linker._nlp = None  # type: ignore[attr-defined]
    retriever.linker.config.embedding_model_name = None
    records = _load_records_or_fallback(args.source, split, limit=max(int(args.limit), 1))
    rows = list(_iter_benchmark_rows(records, retriever))
    if not rows:
        print("No rows available for benchmark.")
        return 1

    summary = {
        "row_count": len(rows),
        "avg_edges_before": mean(row.edges_without_filter for row in rows),
        "avg_edges_after": mean(row.edges_with_filter for row in rows),
        "rows_non_empty_before": sum(1 for row in rows if row.non_empty_without_filter),
        "rows_non_empty_after": sum(1 for row in rows if row.non_empty_with_filter),
        "avg_edges_before_filtering_candidates": mean(row.edges_before_filtering for row in rows),
        "avg_edges_after_filtering_candidates": mean(row.edges_after_filtering for row in rows),
        "avg_node_count_after_filtering": mean(row.node_count_after_filtering for row in rows),
        "avg_gold_edge_retention_proxy_before": mean(row.gold_edge_retention_without_filter for row in rows),
        "avg_gold_edge_retention_proxy_after": mean(row.gold_edge_retention_with_filter for row in rows),
        "fallback_stage_counts": {
            stage: sum(1 for row in rows if row.fallback_stage_used == stage)
            for stage in sorted({row.fallback_stage_used for row in rows})
        },
    }
    payload = {
        "summary": summary,
        "rows": [asdict(row) for row in rows],
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    if args.output_json is not None:
        output_path = args.output_json if args.output_json.is_absolute() else KaggleEnv.path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
