"""Task 1.4 finishing step — fit evidence-weight coefficients on real edges.

Reads `kg/data/calibration_sample.json` (50 PrimeKG edges with annotated
source / evidence_level / n_pubmed_citations), runs the grid search in
kg.evidence_scoring, and persists the chosen coefficients to
`config/evidence_coefficients.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from kg.evidence_scoring import (  # noqa: E402
    DEFAULT_COEFFS_PATH,
    grid_search_coefficients,
)

SAMPLE = REPO_ROOT / "kg" / "data" / "calibration_sample.json"


def main() -> int:
    if not SAMPLE.exists():
        print(f"FATAL: {SAMPLE} missing. Run scripts/validate_primekg_mapping.py first.")
        return 2
    with open(SAMPLE) as f:
        edges = json.load(f)
    print(f"Loaded {len(edges)} edges from {SAMPLE.name}")

    coeffs, stats = grid_search_coefficients(edges)
    print(f"Best coefficients: alpha={coeffs.alpha} beta={coeffs.beta} "
          f"gamma={coeffs.gamma} intercept={coeffs.intercept}")
    print(f"Stats: mean={stats['mean']:.3f}  low_fraction={stats['low_fraction']:.1%}  "
          f"n={stats['n']}")

    DEFAULT_COEFFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_COEFFS_PATH, "w") as f:
        f.write(coeffs.to_json())
    print(f"Wrote {DEFAULT_COEFFS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
