"""Smoke tests for the predicates.yaml schema.

The Phase 6.5 audit found that several predicates referenced in
``kg_build/conflict_detector.CONFLICTING_PREDICATES`` and emitted by
``ingestion/primekg/predicate_map.py`` were NOT declared in
predicates.yaml — silent dead-code in the conflict detector and a
KeyError risk for `interacts_with` lookups. These tests guard the
shipped fix.
"""

from __future__ import annotations

import pytest

from minimed_rag.kg_build.conflict_detector import CONFLICTING_PREDICATES
from minimed_rag.schema_registry.predicate_registry import PredicateRegistry


_NEW_IN_PHASE_6_5 = (
    "interacts_with",
    "causes",
    "prevents",
    "contraindicated_for",
    "increases_risk_of",
    "decreases_risk_of",
)

_PHASE_4_PREDICATES = (
    "treats",
    "causes_adverse_event",
    "associated_with",
    "targets",
    "gene_associated_with_disease",
    "manifested_as",
    "participates_in",
    "involved_in",
)


@pytest.fixture(scope="module")
def registry() -> PredicateRegistry:
    return PredicateRegistry()


@pytest.mark.parametrize("name", _PHASE_4_PREDICATES + _NEW_IN_PHASE_6_5)
def test_every_declared_predicate_loads(registry: PredicateRegistry, name: str) -> None:
    rule = registry.get(name)
    assert rule.name == name
    assert rule.domain, f"{name} has empty domain"
    assert rule.range, f"{name} has empty range"
    assert rule.inverse, f"{name} has empty inverse"


def test_total_predicate_count(registry: PredicateRegistry) -> None:
    """Smoke: 8 Phase-4 + 6 Phase-6.5 = 14. Bumps with each schema add."""
    assert len(registry.all()) == len(_PHASE_4_PREDICATES) + len(_NEW_IN_PHASE_6_5)


def test_conflict_detector_pairs_now_resolve(registry: PredicateRegistry) -> None:
    """Every predicate in CONFLICTING_PREDICATES MUST resolve via the
    registry — otherwise ConflictDetector's named-pair branch silently
    skips real conflicts. This test fails loudly if someone adds a pair
    without declaring the predicate."""
    for left, right in CONFLICTING_PREDICATES:
        assert left in registry.all(), f"CONFLICTING_PREDICATES references undeclared {left}"
        assert right in registry.all(), f"CONFLICTING_PREDICATES references undeclared {right}"


def test_interacts_with_is_symmetric(registry: PredicateRegistry) -> None:
    """drug_drug interactions are commutative; the inverse must be itself."""
    rule = registry.get("interacts_with")
    assert rule.directional is False
    assert rule.inverse == "interacts_with"


def test_causes_drug_disease_valid(registry: PredicateRegistry) -> None:
    assert registry.is_valid_domain_range("causes", "Drug", "Disease") is True


def test_causes_drug_drug_rejected(registry: PredicateRegistry) -> None:
    """`causes` is for adverse outcomes, not drug-drug effects."""
    assert registry.is_valid_domain_range("causes", "Drug", "Drug") is False


def test_prevents_chemical_disease_valid(registry: PredicateRegistry) -> None:
    assert registry.is_valid_domain_range("prevents", "Chemical", "Disease") is True


def test_contraindicated_for_drug_condition_valid(registry: PredicateRegistry) -> None:
    assert registry.is_valid_domain_range("contraindicated_for", "Drug", "Condition") is True


def test_increases_risk_of_exposure_disease_valid(registry: PredicateRegistry) -> None:
    """Exposure is a Phase-6.5-added entity_type; this exercises both the
    new predicate and the new entity_type domain together."""
    assert registry.is_valid_domain_range("increases_risk_of", "Exposure", "Disease") is True


def test_decreases_risk_of_therapy_disease_valid(registry: PredicateRegistry) -> None:
    assert registry.is_valid_domain_range("decreases_risk_of", "Therapy", "Disease") is True
