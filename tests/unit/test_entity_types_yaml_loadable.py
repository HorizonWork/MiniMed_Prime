"""Smoke tests for the entity_types.yaml schema.

Phase 6.5 added 8 entity_types matching what ``ingestion/primekg/normalizer.py``
emits via CamelCase conversion. Without these declarations, the tightened
``predicate_registry.matches_type`` rejects them silently and template-anchored
path traversal drops ~25% of PrimeKG (Gene Ontology + Anatomy + Exposure
subgraphs).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from minimed_rag.common.config import load_yaml


_ENTITY_TYPES_YAML = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "schema"
    / "entity_types.yaml"
)


_PHASE_4_TYPES = {
    "Entity",
    "Drug",
    "Chemical",
    "Disease",
    "Condition",
    "Symptom",
    "Finding",
    "AdverseEvent",
    "Gene",
    "Protein",
    "GeneProtein",
    "Pathway",
    "Procedure",
    "Therapy",
}

_PHASE_6_5_TYPES = {
    "Anatomy",
    "Phenotype",
    "EffectPhenotype",
    "BiologicalProcess",
    "MolecularFunction",
    "CellularComponent",
    "Exposure",
    "Pathogen",
}


@pytest.fixture(scope="module")
def entity_types() -> dict[str, dict]:
    return load_yaml(_ENTITY_TYPES_YAML).get("entity_types", {})


def test_yaml_loads(entity_types: dict[str, dict]) -> None:
    assert entity_types, "entity_types.yaml is empty or missing"


def test_total_count(entity_types: dict[str, dict]) -> None:
    """14 Phase-4 + 8 Phase-6.5 = 22. Bumps with each schema add."""
    assert len(entity_types) == len(_PHASE_4_TYPES) + len(_PHASE_6_5_TYPES)


def test_phase_4_types_still_present(entity_types: dict[str, dict]) -> None:
    missing = _PHASE_4_TYPES - set(entity_types)
    assert not missing, f"Phase-4 entity types regressed: {missing}"


def test_phase_6_5_types_added(entity_types: dict[str, dict]) -> None:
    missing = _PHASE_6_5_TYPES - set(entity_types)
    assert not missing, f"Phase-6.5 entity types not added: {missing}"


def test_parent_chain_resolves(entity_types: dict[str, dict]) -> None:
    """Every parent must exist as a declared type. Catches typos like
    `parent: Disesae` that would silently break inheritance lookups."""
    declared = set(entity_types)
    for name, attrs in entity_types.items():
        parent = attrs.get("parent")
        if parent is None:
            assert name == "Entity", f"{name} has no parent and is not Entity"
            continue
        assert parent in declared, f"{name}.parent={parent} is undeclared"


def test_primekg_camelcased_types_are_declared(entity_types: dict[str, dict]) -> None:
    """The ingestion/primekg/normalizer.py CamelCase-converts raw PrimeKG
    types like ``biological_process`` → ``BiologicalProcess``. Each such
    output must be a declared entity_type — otherwise the tightened
    matches_type drops PrimeKG nodes silently."""
    primekg_outputs = {
        "Drug",
        "Disease",
        "Gene",
        "Protein",
        "GeneProtein",
        "Pathway",
        "Anatomy",
        "Phenotype",
        "EffectPhenotype",
        "BiologicalProcess",
        "MolecularFunction",
        "CellularComponent",
        "Exposure",
    }
    missing = primekg_outputs - set(entity_types)
    assert not missing, f"PrimeKG normalizer emits types not in entity_types.yaml: {missing}"
