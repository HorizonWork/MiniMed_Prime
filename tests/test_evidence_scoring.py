"""Unit tests for kg/evidence_scoring.py."""

from __future__ import annotations

import random

import pytest

from kg.evidence_scoring import (
    Coefficients,
    compute_edge_weight,
    grid_search_coefficients,
)


def test_weight_is_in_unit_interval() -> None:
    edge = {"source": "DrugBank", "evidence_level": "approved", "n_pubmed_citations": 50}
    w = compute_edge_weight(edge)
    assert 0.0 <= w <= 1.0


def test_higher_evidence_yields_higher_weight() -> None:
    high = {"source": "ACC/AHA", "evidence_level": "class_I", "n_pubmed_citations": 200}
    low = {"source": "ConceptNet", "evidence_level": "class_III", "n_pubmed_citations": 0}
    assert compute_edge_weight(high) > compute_edge_weight(low)


def test_unknown_source_uses_default() -> None:
    edge = {"source": "MystérySource", "evidence_level": "approved", "n_pubmed_citations": 0}
    # Should not raise; default authority of 0.5 keeps weight in (0, 1).
    w = compute_edge_weight(edge)
    assert 0.0 < w < 1.0


def test_missing_props_does_not_crash() -> None:
    w = compute_edge_weight({})
    assert 0.0 <= w <= 1.0


def test_grid_search_satisfies_distribution_target() -> None:
    rng = random.Random(0)
    sources = ["ACC/AHA", "DrugBank", "DisGeNET", "ConceptNet"]
    levels = list({
        "approved", "class_I", "class_IIa", "class_IIb", "class_III",
        "score>0.5", "score<0.5", "severe", "moderate", "mild",
    })
    edges = [
        {
            "source": rng.choice(sources),
            "evidence_level": rng.choice(levels),
            "n_pubmed_citations": rng.randint(0, 500),
        }
        for _ in range(50)
    ]

    coeffs, stats = grid_search_coefficients(edges)
    assert 0.4 <= stats["mean"] <= 0.7, stats
    assert stats["low_fraction"] <= 0.05, stats
    assert isinstance(coeffs, Coefficients)


def test_grid_search_raises_on_empty() -> None:
    with pytest.raises(ValueError):
        grid_search_coefficients([])
