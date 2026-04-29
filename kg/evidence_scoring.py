"""Edge evidence weighting for PrimeKG-X (S1 deliverable).

Formula (Final Plan §4.4):

    weight = sigmoid(alpha * evidence_level
                     + beta  * log1p(n_pubmed_citations)
                     + gamma * source_authority
                     - intercept)

`evidence_level` and `source_authority` are normalized to [0, 1] via the
lookup tables below. `n_pubmed_citations` is raw count. The intercept
recenters the distribution; without it, all-positive inputs produce a
right-skewed output that violates the mean ∈ [0.4, 0.7] target.

Coefficients are calibrated by `scripts/calibrate_evidence_weights.py` on a
sample of real PrimeKG edges and persisted to
`config/evidence_coefficients.json`. If that file is absent, the defaults
below (chosen analytically to land near mean = 0.55 on a uniform synthetic
sample) are used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Source-authority ranking (Final Plan §4.4)
SOURCE_AUTHORITY: dict[str, float] = {
    # Tier 1 — clinical guidelines
    "ACC/AHA": 1.0,
    "ACC/AHA-2022": 1.0,
    "ADA-2023": 1.0,
    "NICE": 1.0,
    "GUIDELINE": 1.0,
    # Tier 2 — regulatory / curated drug data
    "DrugBank": 0.8,
    "DrugBank-approved": 0.8,
    # Tier 3 — curated genomic associations
    "DisGeNET": 0.6,
    "DisGeNET-curated": 0.6,
    "HPO": 0.6,
    "GO": 0.6,
    "MONDO": 0.6,
    "Reactome": 0.6,
    "UBERON": 0.6,
    # Tier 4 — co-occurrence / weak evidence
    "ConceptNet": 0.3,
    "ConceptNet-style": 0.3,
}
DEFAULT_AUTHORITY = 0.5

# Evidence-level normalization to [0, 1]
EVIDENCE_LEVEL: dict[str, float] = {
    # FDA / drug status
    "approved": 0.9,
    "investigational": 0.5,
    "experimental": 0.4,
    "vet_approved": 0.6,
    "withdrawn": 0.1,
    # Guideline recommendation classes
    "class_I": 1.0,
    "class_IIa": 0.75,
    "class_IIb": 0.55,
    "class_III": 0.15,
    # DisGeNET buckets
    "score>0.8": 0.85,
    "score>0.5": 0.6,
    "score<0.5": 0.3,
    # DDI severity
    "severe": 0.85,
    "moderate": 0.6,
    "mild": 0.35,
    "contraindicated": 1.0,
    # Frequency buckets
    "very_frequent": 0.9,
    "frequent": 0.65,
    "occasional": 0.35,
    # Generic
    "manual_review": 0.85,
    "computational": 0.45,
    "automatic": 0.3,
}
DEFAULT_LEVEL = 0.5


@dataclass(frozen=True)
class Coefficients:
    """Tuned coefficients for compute_edge_weight."""

    alpha: float = 0.9     # evidence_level multiplier
    beta: float = 0.15     # log1p(n_pubmed) multiplier
    gamma: float = 0.7     # source_authority multiplier
    intercept: float = 1.0 # recenters distribution

    def to_json(self) -> str:
        return json.dumps(
            {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma, "intercept": self.intercept},
            indent=2,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Coefficients":
        with open(path) as f:
            d = json.load(f)
        return cls(alpha=d["alpha"], beta=d["beta"], gamma=d["gamma"], intercept=d["intercept"])


_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COEFFS_PATH = _REPO_ROOT / "config" / "evidence_coefficients.json"


def _load_default_coefficients() -> Coefficients:
    if DEFAULT_COEFFS_PATH.exists():
        return Coefficients.from_file(DEFAULT_COEFFS_PATH)
    return Coefficients()


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def compute_edge_weight(
    edge_props: dict[str, Any],
    coeffs: Coefficients | None = None,
) -> float:
    """Compute the [0, 1] edge weight from an edge's properties.

    Required keys: at least one of `evidence_level` or `source` (others
    fall back to defaults). `n_pubmed_citations` defaults to 0.
    """
    if coeffs is None:
        coeffs = _load_default_coefficients()

    level_key = edge_props.get("evidence_level") or edge_props.get("level") or ""
    level = EVIDENCE_LEVEL.get(level_key, DEFAULT_LEVEL)

    src_key = edge_props.get("source") or edge_props.get("src") or ""
    authority = SOURCE_AUTHORITY.get(src_key, DEFAULT_AUTHORITY)

    n_cites = edge_props.get("n_pubmed_citations") or 0
    log_cites = math.log1p(max(n_cites, 0))

    x = (
        coeffs.alpha * level
        + coeffs.beta * log_cites
        + coeffs.gamma * authority
        - coeffs.intercept
    )
    return _sigmoid(x)


def grid_search_coefficients(
    edges: list[dict[str, Any]],
    target_mean_lo: float = 0.4,
    target_mean_hi: float = 0.7,
    max_low_fraction: float = 0.05,
    low_threshold: float = 0.1,
) -> tuple[Coefficients, dict[str, float]]:
    """Grid-search (alpha, beta, gamma, intercept) on a sample of edges.

    Returns the best coefficients (closest to mean=0.55, satisfying the
    low-fraction constraint) plus the diagnostic stats for that choice.
    """
    if not edges:
        raise ValueError("edges sample is empty")

    target_center = (target_mean_lo + target_mean_hi) / 2.0
    best: tuple[Coefficients, dict[str, float], float] | None = None

    alpha_grid = [0.4, 0.6, 0.8, 0.9, 1.0, 1.2, 1.5]
    beta_grid = [0.05, 0.10, 0.15, 0.20, 0.30]
    gamma_grid = [0.4, 0.6, 0.8, 1.0]
    intercept_grid = [0.6, 0.8, 1.0, 1.2, 1.4]

    for a in alpha_grid:
        for b in beta_grid:
            for g in gamma_grid:
                for i in intercept_grid:
                    c = Coefficients(alpha=a, beta=b, gamma=g, intercept=i)
                    weights = [compute_edge_weight(e, c) for e in edges]
                    n = len(weights)
                    mean = sum(weights) / n
                    low_frac = sum(1 for w in weights if w < low_threshold) / n
                    if low_frac > max_low_fraction:
                        continue
                    if not (target_mean_lo <= mean <= target_mean_hi):
                        continue
                    score = abs(mean - target_center)
                    if best is None or score < best[2]:
                        stats = {"mean": mean, "low_fraction": low_frac, "n": n}
                        best = (c, stats, score)

    if best is None:
        raise RuntimeError(
            "no coefficients in the grid satisfied "
            f"mean ∈ [{target_mean_lo}, {target_mean_hi}] with low_fraction ≤ {max_low_fraction}"
        )
    return best[0], best[1]
