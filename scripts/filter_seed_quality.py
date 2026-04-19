#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import DEFAULT_GOLDISH_BACKEND_PATTERNS, filter_seed_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Split prepared MedReason seed JSONL into goldish/silver/reject tiers.")
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Prepared seed JSONL from scripts/prepare_medreason_seed.py.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write tiered JSONL outputs and summary.")
    parser.add_argument("--goldish-name", default="trm_seed_goldish.jsonl")
    parser.add_argument("--silver-name", default="trm_seed_silver.jsonl")
    parser.add_argument("--reject-name", default="trm_seed_rejects.jsonl")
    parser.add_argument("--summary-name", default="seed_quality_summary.json")
    parser.add_argument("--min-gold-edges", type=int, default=1)
    parser.add_argument(
        "--accepted-goldish-edge-backends",
        default=",".join(DEFAULT_GOLDISH_BACKEND_PATTERNS),
        help="Comma-separated backend patterns allowed into goldish tier. Supports '*' suffix wildcards, e.g. direct,llm_*.",
    )
    parser.add_argument("--allow-rule-based-entity-linker", action="store_true", help="Do not demote rule-based entity linker runs to silver.")
    parser.add_argument("--allow-unknown-primekg-backend", action="store_true", help="Do not demote records with missing PrimeKG backend metadata.")
    args = parser.parse_args()

    accepted_patterns = tuple(pattern.strip() for pattern in args.accepted_goldish_edge_backends.split(",") if pattern.strip())
    summary = filter_seed_jsonl(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        goldish_name=args.goldish_name,
        silver_name=args.silver_name,
        reject_name=args.reject_name,
        summary_name=args.summary_name,
        min_gold_edges=args.min_gold_edges,
        accepted_goldish_edge_backends=accepted_patterns or DEFAULT_GOLDISH_BACKEND_PATTERNS,
        require_real_entity_linker=not args.allow_rule_based_entity_linker,
        require_real_primekg=not args.allow_unknown_primekg_backend,
    )
    sys.stdout.write(json.dumps(summary.to_json_dict(), indent=2, sort_keys=True) + "\n")
    return 0 if summary.goldish_records > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
