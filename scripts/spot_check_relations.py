"""Task 1.1b — Sample MedReason adjacency pairs for relation-gap spot-check.

Extracts ~50 random (entity_a -> entity_b) pairs from MedReason gold chains,
annotates each with the heuristic entity-type classification from
analyze_medreason_coverage.py, and writes them to
`reports/medreason_pairs_sample.json` for LLM-side classification.

The companion file `reports/medreason_relation_spot_check.md` is then
populated by reading these pairs and assigning each to one of the 40
schema relations (or `no_match`). The output is a sanity check on whether
the schema has any large semantic gaps the entity/pair coverage misses.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset  # noqa: E402

# Re-use the heuristic classifier and chain extractor from the main script.
from scripts.analyze_medreason_coverage import (  # noqa: E402
    classify_entity,
    extract_chains,
)

SAMPLE_SIZE = 50
RECORD_SAMPLE = 300   # scan more records to get enough chains
SEED = 1
OUT = REPO_ROOT / "reports" / "medreason_pairs_sample.json"


def main() -> int:
    print("Loading MedReason…")
    ds = load_dataset("UCSC-VLAA/MedReason", split="train")
    print(f"  {len(ds):,} rows")

    rng = random.Random(SEED)
    indices = rng.sample(range(len(ds)), min(RECORD_SAMPLE, len(ds)))

    all_pairs: list[dict] = []
    for i in indices:
        rec = ds[i]
        chains = extract_chains(rec.get("reasoning", "") or "")
        for chain in chains:
            for a, b in zip(chain, chain[1:]):
                ta = classify_entity(a)
                tb = classify_entity(b)
                if ta is None or tb is None:
                    continue
                all_pairs.append({
                    "source_dataset": rec.get("dataset_name", "?"),
                    "question": (rec.get("question") or "")[:160],
                    "entity_a": a,
                    "entity_a_type": ta,
                    "entity_b": b,
                    "entity_b_type": tb,
                    "inferred_relation": None,   # filled in by LLM (Claude)
                    "reasoning": "",
                })

    print(f"  collected {len(all_pairs):,} type-classified pairs")
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(all_pairs)
    sample = all_pairs[:SAMPLE_SIZE]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUT.relative_to(REPO_ROOT)} ({len(sample)} pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
