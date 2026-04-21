#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.layers.layer1_retrieval import AgenticRetriever
from src.training import DEFAULT_MEDREASON_SOURCE, prepare_medreason_seed_jsonl
from src.utils.kaggle_env import KaggleEnv, T4Hardening


def _parse_relation_filter_overrides(values: list[str]) -> dict[str, set[str]] | None:
    overrides: dict[str, set[str]] = {}
    for raw_value in values:
        question_type, separator, relations = raw_value.partition("=")
        if not separator:
            raise ValueError(f"Invalid --relation-filter-override value: {raw_value!r}")
        parsed_relations = {relation.strip() for relation in relations.split(",") if relation.strip()}
        if not question_type.strip() or not parsed_relations:
            raise ValueError(f"Invalid --relation-filter-override value: {raw_value!r}")
        overrides[question_type.strip()] = parsed_relations
    return overrides or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare TRM seed JSONL from raw MedReason records.")
    parser.add_argument("--source", default=DEFAULT_MEDREASON_SOURCE, help="HF dataset name or local MedReason file/dir.")
    parser.add_argument("--split", default="train", help="Dataset split name; use 'none' for local unsplit files.")
    parser.add_argument("--output-jsonl", type=Path, default=Path("data/trm_seed.jsonl"), help="Output normalized seed JSONL.")
    parser.add_argument("--limit", type=int, default=None, help="Max raw records to read.")
    parser.add_argument("--max-saved", type=int, default=None, help="Stop after this many prepared examples.")
    parser.add_argument("--edge-mapper", choices=("auto", "direct", "heuristic", "llm"), default="auto")
    parser.add_argument(
        "--llm-model-name",
        default=os.getenv("MEDREASON_EDGE_LLM", "heuristic"),
        help="Judge/LLM model for --edge-mapper llm. With --edge-mapper auto, OPENAI_API_KEY selects gpt-4o-mini, then GEMINI_API_KEY selects gemini-2.5-flash.",
    )
    parser.add_argument("--llm-device", default="cpu")
    parser.add_argument("--allow-empty-gold", action="store_true", help="Write examples even when no gold edge maps.")
    parser.add_argument("--primekg-path", type=Path, default=Path("data/kg/primekg"))
    parser.add_argument("--pubmed-cache-path", type=Path, default=Path("data/pubmed_cache.jsonl"))
    parser.add_argument("--pubmed-api-key", default=None)
    parser.add_argument("--relation-filter", default=None, help="Comma-separated PrimeKG relations to keep.")
    parser.add_argument("--disable-relation-filter", action="store_true", help="Disable question-type-aware relation filtering.")
    parser.add_argument(
        "--relation-filter-override",
        action="append",
        default=[],
        help="Repeatable question_type=rel1,rel2 override that replaces the routed relation set for one question type.",
    )
    parser.add_argument("--kaggle", action="store_true", help="Enable Kaggle path and T4 memory handling.")
    args = parser.parse_args()

    if args.kaggle or KaggleEnv.is_kaggle():
        T4Hardening.setup_memory()

    split = None if args.split.lower() in {"", "none", "null", "false"} else args.split
    relation_filter = (
        {relation.strip() for relation in args.relation_filter.split(",") if relation.strip()}
        if args.relation_filter
        else None
    )
    relation_filter_overrides = _parse_relation_filter_overrides(args.relation_filter_override)
    retriever = AgenticRetriever(
        primekg_path=KaggleEnv.path(args.primekg_path),
        pubmed_api_key=args.pubmed_api_key,
        pubmed_cache_path=KaggleEnv.ensure_writeable(KaggleEnv.path(args.pubmed_cache_path)),
        use_relation_filter=not args.disable_relation_filter,
        relation_filter=relation_filter,
        relation_filter_overrides=relation_filter_overrides,
    )
    result = prepare_medreason_seed_jsonl(
        source=args.source,
        output_jsonl=args.output_jsonl,
        retriever=retriever,
        split=split,
        limit=args.limit,
        max_saved=args.max_saved,
        edge_mapper_backend=args.edge_mapper,
        llm_model_name=args.llm_model_name,
        llm_device=args.llm_device,
        allow_empty_gold=args.allow_empty_gold,
    )
    sys.stdout.write(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n")
    return 0 if (result.records_saved > 0 or result.records_skipped > 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
