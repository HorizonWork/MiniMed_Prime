#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.primekg_grounding import node_lookup_keys, normalize_basic_text  # noqa: E402
from src.utils.kaggle_env import KaggleEnv  # noqa: E402

DEFAULT_TERMS = (
    "H pylori",
    "Colle's fascia",
    "common hepatic aery",
    "right gastroepiploic aery",
    "Typhoid",
    "urogenital diaphragm",
)


def analyze_terms(nodes_csv: Path, terms: list[str]) -> dict[str, Any]:
    with nodes_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        lower_header = {column.lower() for column in header}
        node_id_column = _first_present(header, ("node_id", "id", "primekg_node_id")) or "node_id"
        node_name_column = _first_present(header, ("node_name", "name")) or "node_name"
        node_type_column = _first_present(header, ("node_type", "type")) or "node_type"

        term_results: dict[str, dict[str, Any]] = {
            term: {
                "term": term,
                "basic_key": normalize_basic_text(term),
                "lookup_keys": sorted(node_lookup_keys(term)),
                "matches": [],
            }
            for term in terms
        }

        for row in reader:
            node_name = str(row.get(node_name_column, "") or "")
            node_keys = node_lookup_keys(node_name)
            for term, result in term_results.items():
                if not set(result["lookup_keys"]) & node_keys:
                    continue
                result["matches"].append(
                    {
                        "node_id": str(row.get(node_id_column, "") or ""),
                        "node_name": node_name,
                        "node_type": str(row.get(node_type_column, "") or ""),
                    }
                )

    analysis = {
        "nodes_csv": str(nodes_csv),
        "header": header,
        "has_cui_columns": any(column in lower_header for column in {"node_cui", "cui", "umls_cui", "umls_id"}),
        "terms": [],
    }
    for term in terms:
        matches = term_results[term]["matches"]
        analysis["terms"].append(
            {
                **term_results[term],
                "match_count": len(matches),
                "matched": bool(matches),
                "matches": matches[:10],
            }
        )
    return analysis


def _first_present(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lightweight PrimeKG grounding analysis without building a graph.")
    parser.add_argument(
        "--nodes-csv",
        type=Path,
        default=KaggleEnv.path("data/kg/primekg/nodes.csv"),
        help="Path to PrimeKG nodes.csv",
    )
    parser.add_argument(
        "--term",
        action="append",
        default=[],
        help="Term to probe against node names. Repeat for multiple terms.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    terms = list(args.term or DEFAULT_TERMS)
    payload = analyze_terms(args.nodes_csv, terms)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
