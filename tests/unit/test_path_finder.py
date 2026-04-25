"""Unit tests for the Phase 6 PathFinder.

Uses an in-memory fake Neo4j that returns canned rows for a given Cypher
query, simulating the reified-Assertion KG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minimed_rag.reasoning.path_finder import PathFinder
from minimed_rag.schema_registry.metapath_registry import (
    MetapathRegistry,
    MetapathStep,
    MetapathTemplate,
)
from minimed_rag.schema_registry.predicate_registry import PredicateRegistry


@dataclass
class _PredicateRule:
    name: str
    domain: list[str]
    range: list[str]
    directional: bool
    inverse: str


class _FakePredicateRegistry:
    def __init__(self) -> None:
        self.rules: dict[str, _PredicateRule] = {
            "treats": _PredicateRule(
                "treats", ["Drug"], ["Disease"], True, "treated_by"
            ),
            "targets": _PredicateRule(
                "targets", ["Drug"], ["Gene", "Protein"], True, "target_of"
            ),
            "involved_in": _PredicateRule(
                "involved_in", ["Pathway"], ["Disease"], True, "involves_pathway"
            ),
            "gene_associated_with_disease": _PredicateRule(
                "gene_associated_with_disease",
                ["Gene", "Protein"],
                ["Disease"],
                False,
                "disease_associated_with_gene",
            ),
        }

    def get(self, name: str) -> _PredicateRule:
        if name not in self.rules:
            raise KeyError(name)
        return self.rules[name]

    def all(self) -> dict[str, _PredicateRule]:
        return self.rules

    def is_directional(self, name: str) -> bool:
        return self.get(name).directional

    @staticmethod
    def matches_type(entity_type: str | None, allowed: list[str]) -> bool:
        return PredicateRegistry.matches_type(entity_type, allowed)


@dataclass
class _FakeNeo4j:
    last_cypher: str = ""
    last_params: dict | None = None
    rows: list[dict] | None = None

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        self.last_cypher = cypher
        self.last_params = params or {}
        return list(self.rows or [])


def _registry_with(template: MetapathTemplate) -> MetapathRegistry:
    reg = MetapathRegistry.__new__(MetapathRegistry)
    reg.templates = {template.name: template}
    return reg


def _template(name: str, task_type: str, pattern: list[tuple[str, str, str]],
              priority: float = 1.0, min_conf: float = 0.5) -> MetapathTemplate:
    return MetapathTemplate(
        name=name,
        task_type=task_type,
        pattern=[MetapathStep(*step) for step in pattern],
        max_depth=len(pattern),
        priority=priority,
        min_edge_confidence=min_conf,
    )


def test_one_hop_directional_forward_path() -> None:
    template = _template("treatment_direct", "treatment", [("Drug", "treats", "Disease")])
    reg = _registry_with(template)
    pred_reg = _FakePredicateRegistry()
    rows = [
        {
            "n0": {"entity_id": "KG:Drug:m"},
            "a1": {"assertion_id": "A1", "predicate": "treats", "confidence": 0.9},
            "n1": {"entity_id": "KG:Disease:d"},
            "r1a": "SUBJECT",
            "r1b": "OBJECT",
            "l0": ["Entity"],
            "l1": ["Entity"],
        }
    ]
    neo4j = _FakeNeo4j(rows=rows)
    finder = PathFinder(neo4j, reg, pred_reg)

    paths = finder.find_paths_from_anchor("KG:Drug:m", "treatment")

    assert len(paths) == 1
    p = paths[0]
    assert p.start_entity_id == "KG:Drug:m"
    assert p.end_entity_id == "KG:Disease:d"
    assert p.metapath_template_id == "treatment_direct"
    assert len(p.steps) == 1
    s = p.steps[0]
    assert s.predicate == "treats"
    assert s.assertion_id == "A1"
    assert s.edge_confidence == 0.9
    assert s.direction == "forward"


def test_one_hop_template_inverse_decoded() -> None:
    # Template (Disease, involved_in, Pathway) but predicate's domain is Pathway.
    template = _template(
        "disease_pathway_inverse", "tx", [("Disease", "involved_in", "Pathway")]
    )
    reg = _registry_with(template)
    pred_reg = _FakePredicateRegistry()
    rows = [
        {
            "n0": {"entity_id": "D:1"},
            "a1": {"assertion_id": "A1", "predicate": "involved_in", "confidence": 0.8},
            "n1": {"entity_id": "P:1"},
            "r1a": "OBJECT",
            "r1b": "SUBJECT",
            "l0": ["Entity"],
            "l1": ["Entity"],
        }
    ]
    finder = PathFinder(_FakeNeo4j(rows=rows), reg, pred_reg)
    paths = finder.find_paths_from_anchor("D:1", "tx")

    assert len(paths) == 1
    assert paths[0].steps[0].direction == "template_inverse"


def test_non_directional_predicate_marks_forward() -> None:
    template = _template(
        "gd_direct", "gene_disease",
        [("Disease", "gene_associated_with_disease", "Gene")],
    )
    reg = _registry_with(template)
    pred_reg = _FakePredicateRegistry()
    rows = [
        {
            "n0": {"entity_id": "D:1"},
            "a1": {
                "assertion_id": "A1",
                "predicate": "gene_associated_with_disease",
                "confidence": 0.7,
            },
            "n1": {"entity_id": "G:1"},
            "r1a": "OBJECT",
            "r1b": "SUBJECT",
            "l0": ["Entity"],
            "l1": ["Entity"],
        }
    ]
    finder = PathFinder(_FakeNeo4j(rows=rows), reg, pred_reg)
    paths = finder.find_paths_from_anchor("D:1", "gene_disease")
    assert paths[0].steps[0].direction == "forward"


def test_two_hop_path_built_with_step_indices() -> None:
    template = _template(
        "tx_mech", "tx",
        [("Drug", "targets", "Gene"), ("Gene", "gene_associated_with_disease", "Disease")],
    )
    reg = _registry_with(template)
    pred_reg = _FakePredicateRegistry()
    rows = [
        {
            "n0": {"entity_id": "Dr:1"},
            "a1": {"assertion_id": "A1", "predicate": "targets", "confidence": 0.8},
            "n1": {"entity_id": "G:1"},
            "a2": {
                "assertion_id": "A2",
                "predicate": "gene_associated_with_disease",
                "confidence": 0.6,
            },
            "n2": {"entity_id": "D:1"},
            "r1a": "SUBJECT",
            "r1b": "OBJECT",
            "r2a": "SUBJECT",
            "r2b": "OBJECT",
            "l0": ["Entity"],
            "l1": ["Entity"],
            "l2": ["Entity"],
        }
    ]
    finder = PathFinder(_FakeNeo4j(rows=rows), reg, pred_reg)
    paths = finder.find_paths_from_anchor("Dr:1", "tx")
    assert len(paths) == 1
    p = paths[0]
    assert len(p.steps) == 2
    assert [s.step_index for s in p.steps] == [0, 1]
    assert p.start_entity_id == "Dr:1"
    assert p.end_entity_id == "D:1"


def test_concept_anchor_falls_back_to_cui() -> None:
    template = _template("tx_dir", "tx", [("Drug", "treats", "Disease")])
    reg = _registry_with(template)
    pred_reg = _FakePredicateRegistry()
    rows = [
        {
            "n0": {"cui": "C0001"},
            "a1": {"assertion_id": "A1", "predicate": "treats", "confidence": 0.8},
            "n1": {"entity_id": "D:1"},
            "r1a": "SUBJECT",
            "r1b": "OBJECT",
            "l0": ["Concept"],
            "l1": ["Entity"],
        }
    ]
    finder = PathFinder(_FakeNeo4j(rows=rows), reg, pred_reg)
    paths = finder.find_paths_from_anchor("C0001", "tx")
    assert paths[0].start_entity_id == "C0001"


def test_template_free_paths_have_empty_template_id() -> None:
    pred_reg = _FakePredicateRegistry()
    reg = MetapathRegistry.__new__(MetapathRegistry)
    reg.templates = {}
    rows = [
        {
            "n0": {"entity_id": "A"},
            "a1": {"assertion_id": "A1", "predicate": "treats", "confidence": 0.8},
            "n1": {"entity_id": "B"},
            "r1a": "SUBJECT",
            "r1b": "OBJECT",
            "l0": ["Entity"],
            "l1": ["Entity"],
        }
    ]
    finder = PathFinder(_FakeNeo4j(rows=rows), reg, pred_reg)
    paths = finder.find_template_free_paths("A", max_depth=1)
    assert len(paths) == 1
    assert paths[0].metapath_template_id == ""
    assert paths[0].task_type == "general"


def test_empty_anchor_returns_empty() -> None:
    pred_reg = _FakePredicateRegistry()
    reg = MetapathRegistry.__new__(MetapathRegistry)
    reg.templates = {}
    finder = PathFinder(_FakeNeo4j(rows=[]), reg, pred_reg)
    assert finder.find_paths_from_anchor("", "tx") == []
    assert finder.find_template_free_paths("") == []


def test_neo4j_exception_swallowed_per_template() -> None:
    template = _template("tx_dir", "tx", [("Drug", "treats", "Disease")])
    reg = _registry_with(template)
    pred_reg = _FakePredicateRegistry()

    class _BoomNeo4j:
        def query(self, *_args, **_kwargs):
            raise RuntimeError("simulated neo4j error")

    finder = PathFinder(_BoomNeo4j(), reg, pred_reg)
    assert finder.find_paths_from_anchor("X", "tx") == []
