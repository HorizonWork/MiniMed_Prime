"""Unit tests for the runtime GraphClient."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from minimed_rag.kg.assertion_validator import AssertionDirectionValidator
from minimed_rag.kg.graph_client import (
    FIND_CONCEPT_BY_CUI_CYPHER,
    FIND_ENTITY_BY_CUI_CYPHER,
    FIND_ENTITY_BY_NAME_CYPHER,
    GET_NEIGHBORS_DEPTH_1_CYPHER,
    GET_TWO_HOP_PATHS_CYPHER,
    GraphClient,
)


@dataclass
class FakeNeo4j:
    responses: dict[str, list[dict]] = field(default_factory=dict)
    calls: list[tuple[str, dict]] = field(default_factory=list)

    def query(self, cypher: str, params: dict) -> list[dict]:
        self.calls.append((cypher, params))
        for key, rows in self.responses.items():
            if key in cypher:
                return rows
        return []


class FakePredicateRegistry:
    def __init__(self, valid: set[tuple[str, str, str]], predicates: set[str] | None = None):
        self.valid = valid
        self.rules = {predicate: object() for predicate in (predicates or set())}
        self.rules.update({predicate: object() for predicate, _, _ in valid})

    def all(self) -> dict[str, object]:
        return self.rules

    def is_valid_domain_range(self, predicate: str, subject_type: str, object_type: str) -> bool:
        return (predicate, subject_type, object_type) in self.valid


def test_get_neighbors_depth_1_builds_expected_cypher():
    neo4j = FakeNeo4j(
        responses={
            "get_neighbors_hint": [],
            "MATCH (anchor)": [
                {
                    "neighbor": {
                        "entity_id": "KG:Disease:abc",
                        "preferred_name": "Type 2 diabetes",
                        "entity_type": "Disease",
                    },
                    "neighbor_labels": ["Entity"],
                    "assertion": {
                        "assertion_id": "ASSERT:1",
                        "predicate": "treats",
                        "confidence": 0.9,
                        "polarity": "positive",
                        "source_system": "primekg",
                    },
                    "role_in": "SUBJECT",
                    "role_out": "OBJECT",
                }
            ],
        }
    )
    client = GraphClient(neo4j=neo4j)
    neighbors = client.get_neighbors("C0025598", depth=1, limit=10, min_confidence=0.3)

    cypher, params = neo4j.calls[0]
    assert "MATCH (anchor)" in cypher
    assert "a.is_current = true" in cypher
    assert "a.confidence >= $min_conf" in cypher
    assert "a.predicate = 'treats'" not in cypher
    assert "subject, object" in cypher
    assert "labels(subject) AS subject_labels" in cypher
    assert "role_in:SUBJECT|OBJECT" in cypher
    assert params == {"key": "C0025598", "limit": 10, "min_conf": 0.3}
    assert cypher == GET_NEIGHBORS_DEPTH_1_CYPHER

    assert len(neighbors) == 1
    n = neighbors[0]
    assert n.node.entity_id == "KG:Disease:abc"
    assert n.node.preferred_name == "Type 2 diabetes"
    assert n.predicate == "treats"
    assert n.confidence == 0.9
    assert n.polarity == "positive"
    assert n.direction == "out"


def test_get_neighbors_anchor_as_object_marks_incoming_direction():
    neo4j = FakeNeo4j(
        responses={
            "MATCH (anchor)": [
                {
                    "neighbor": {
                        "entity_id": "KG:Drug:m",
                        "preferred_name": "Metformin",
                        "entity_type": "Drug",
                    },
                    "neighbor_labels": ["Entity"],
                    "assertion": {
                        "assertion_id": "ASSERT:1",
                        "predicate": "treats",
                        "confidence": 0.9,
                        "polarity": "positive",
                        "source_system": "primekg",
                    },
                    "role_in": "OBJECT",
                    "role_out": "SUBJECT",
                }
            ],
        }
    )
    client = GraphClient(neo4j=neo4j)
    neighbors = client.get_neighbors("KG:Disease:d", depth=1)

    assert neighbors[0].direction == "in"


def test_get_neighbors_filters_invalid_assertion_with_schema_validator():
    neo4j = FakeNeo4j(
        responses={
            "MATCH (anchor)": [
                {
                    "neighbor": {
                        "entity_id": "KG:Drug:m",
                        "preferred_name": "Drug X",
                        "entity_type": "Drug",
                    },
                    "neighbor_labels": ["Entity"],
                    "subject": {
                        "entity_id": "KG:Finding:f",
                        "preferred_name": "Finding X",
                        "entity_type": "Finding",
                    },
                    "subject_labels": ["Entity"],
                    "object": {
                        "entity_id": "KG:Drug:m",
                        "preferred_name": "Drug X",
                        "entity_type": "Drug",
                    },
                    "object_labels": ["Entity"],
                    "assertion": {
                        "assertion_id": "ASSERT:bad",
                        "predicate": "causes_adverse_event",
                        "confidence": 0.9,
                        "polarity": "positive",
                        "source_system": "primekg",
                    },
                    "role_in": "SUBJECT",
                    "role_out": "OBJECT",
                }
            ],
        }
    )
    validator = AssertionDirectionValidator(
        FakePredicateRegistry(
            valid={("causes_adverse_event", "Drug", "AdverseEvent")},
            predicates={"causes_adverse_event"},
        )
    )
    client = GraphClient(neo4j=neo4j, assertion_validator=validator)

    assert client.get_neighbors("KG:Finding:f", depth=1) == []


def test_get_two_hop_paths_preserves_assertion_roles():
    neo4j = FakeNeo4j(
        responses={
            "role_anchor_1": [
                {
                    "anchor": {"entity_id": "KG:Drug:m", "preferred_name": "Metformin"},
                    "anchor_labels": ["Entity"],
                    "mid": {"entity_id": "KG:Gene:g", "preferred_name": "AMPK"},
                    "mid_labels": ["Entity"],
                    "terminal": {"entity_id": "KG:Disease:d", "preferred_name": "T2D"},
                    "terminal_labels": ["Entity"],
                    "edge1": {
                        "assertion_id": "A1",
                        "predicate": "targets",
                        "confidence": 0.8,
                        "polarity": "positive",
                        "source_system": "primekg",
                    },
                    "edge2": {
                        "assertion_id": "A2",
                        "predicate": "associated_with",
                        "confidence": 0.7,
                        "polarity": "positive",
                        "source_system": "primekg",
                    },
                    "role_anchor_1": "SUBJECT",
                    "role_mid_1": "OBJECT",
                    "role_mid_2": "OBJECT",
                    "role_terminal_2": "SUBJECT",
                }
            ]
        }
    )
    client = GraphClient(neo4j=neo4j)
    paths = client.get_two_hop_paths("KG:Drug:m", limit=3, min_confidence=0.5)

    cypher, params = neo4j.calls[0]
    assert cypher == GET_TWO_HOP_PATHS_CYPHER
    assert params == {"key": "KG:Drug:m", "limit": 3, "min_conf": 0.5}
    assert paths[0].edge1.subject_key == "KG:Drug:m"
    assert paths[0].edge1.object_key == "KG:Gene:g"
    assert paths[0].edge2.subject_key == "KG:Disease:d"
    assert paths[0].edge2.object_key == "KG:Gene:g"


def test_get_two_hop_paths_filters_invalid_edge_with_schema_validator():
    neo4j = FakeNeo4j(
        responses={
            "role_anchor_1": [
                {
                    "anchor": {
                        "entity_id": "KG:Drug:m",
                        "preferred_name": "Metformin",
                        "entity_type": "Drug",
                    },
                    "anchor_labels": ["Entity"],
                    "mid": {
                        "entity_id": "KG:Gene:g",
                        "preferred_name": "AMPK",
                        "entity_type": "Gene",
                    },
                    "mid_labels": ["Entity"],
                    "terminal": {
                        "entity_id": "KG:Disease:d",
                        "preferred_name": "T2D",
                        "entity_type": "Disease",
                    },
                    "terminal_labels": ["Entity"],
                    "subject1": {"entity_id": "KG:Drug:m", "entity_type": "Drug"},
                    "subject1_labels": ["Entity"],
                    "object1": {"entity_id": "KG:Gene:g", "entity_type": "Gene"},
                    "object1_labels": ["Entity"],
                    "subject2": {"entity_id": "KG:Disease:d", "entity_type": "Disease"},
                    "subject2_labels": ["Entity"],
                    "object2": {"entity_id": "KG:Drug:m", "entity_type": "Drug"},
                    "object2_labels": ["Entity"],
                    "edge1": {
                        "assertion_id": "A1",
                        "predicate": "targets",
                        "confidence": 0.8,
                        "polarity": "positive",
                        "source_system": "primekg",
                    },
                    "edge2": {
                        "assertion_id": "A2",
                        "predicate": "causes_adverse_event",
                        "confidence": 0.7,
                        "polarity": "positive",
                        "source_system": "primekg",
                    },
                    "role_anchor_1": "SUBJECT",
                    "role_mid_1": "OBJECT",
                    "role_mid_2": "SUBJECT",
                    "role_terminal_2": "OBJECT",
                }
            ]
        }
    )
    validator = AssertionDirectionValidator(
        FakePredicateRegistry(
            valid={
                ("targets", "Drug", "Gene"),
                ("causes_adverse_event", "Drug", "AdverseEvent"),
            },
            predicates={"targets", "causes_adverse_event"},
        )
    )
    client = GraphClient(neo4j=neo4j, assertion_validator=validator)

    assert client.get_two_hop_paths("KG:Drug:m") == []


def test_get_neighbors_raises_on_multi_hop():
    client = GraphClient(neo4j=FakeNeo4j())
    with pytest.raises(NotImplementedError, match="depth=2"):
        client.get_neighbors("X", depth=2)


def test_get_path_returns_nodes_in_order():
    neo4j = FakeNeo4j(
        responses={
            "shortestPath": [
                {
                    "nodes": [
                        {
                            "labels": ["Entity"],
                            "props": {
                                "entity_id": "KG:Drug:a",
                                "preferred_name": "Metformin",
                            },
                        },
                        {
                            "labels": ["Assertion"],
                            "props": {"assertion_id": "ASSERT:1"},
                        },
                        {
                            "labels": ["Entity"],
                            "props": {
                                "entity_id": "KG:Disease:b",
                                "preferred_name": "Diabetes",
                            },
                        },
                    ],
                    "rel_types": ["SUBJECT", "OBJECT"],
                    "path_len": 2,
                }
            ]
        }
    )
    client = GraphClient(neo4j=neo4j)
    nodes = client.get_path("C0025598", "C0011849", max_depth=4)

    cypher, params = neo4j.calls[0]
    assert "shortestPath" in cypher
    assert "*..4" in cypher
    assert params == {"cui_a": "C0025598", "cui_b": "C0011849"}
    assert [n.preferred_name for n in nodes] == ["Metformin", None, "Diabetes"]


def test_get_path_empty_when_no_results():
    client = GraphClient(neo4j=FakeNeo4j())
    assert client.get_path("A", "B") == []


def test_get_path_rejects_invalid_depth():
    client = GraphClient(neo4j=FakeNeo4j())
    with pytest.raises(ValueError):
        client.get_path("A", "B", max_depth=0)
    with pytest.raises(ValueError):
        client.get_path("A", "B", max_depth=50)


def test_find_entity_by_cui_uses_canonical_cui_index():
    neo4j = FakeNeo4j(
        responses={
            "WHERE e.canonical_cui = $cui": [
                {
                    "node": {"entity_id": "KG:Drug:m", "canonical_cui": "C0025598"},
                    "node_labels": ["Entity"],
                }
            ]
        }
    )
    client = GraphClient(neo4j=neo4j)
    results = client.find_entity_by_cui("C0025598", limit=3)

    cypher, params = neo4j.calls[0]
    assert cypher == FIND_ENTITY_BY_CUI_CYPHER
    assert params == {"cui": "C0025598", "limit": 3}
    assert len(results) == 1
    assert results[0].canonical_cui == "C0025598"


def test_find_entity_by_name_uses_fulltext_index():
    neo4j = FakeNeo4j(
        responses={
            "db.index.fulltext.queryNodes": [
                {
                    "node": {"entity_id": "KG:Drug:m", "preferred_name": "Metformin"},
                    "node_labels": ["Entity"],
                    "score": 1.0,
                }
            ]
        }
    )
    client = GraphClient(neo4j=neo4j)
    results = client.find_entity_by_name("metformin", limit=5)
    assert results[0].preferred_name == "Metformin"
    cypher, _ = neo4j.calls[0]
    assert "entity_name_ft" in cypher
    assert cypher == FIND_ENTITY_BY_NAME_CYPHER


def test_find_concept_by_cui_returns_none_when_missing():
    client = GraphClient(neo4j=FakeNeo4j())
    assert client.find_concept_by_cui("C_missing") is None


def test_find_concept_by_cui_returns_node_when_present():
    neo4j = FakeNeo4j(
        responses={
            FIND_CONCEPT_BY_CUI_CYPHER: [
                {
                    "node": {
                        "cui": "C0025598",
                        "preferred_name": "Metformin",
                        "semantic_types": ["Pharmacologic Substance"],
                    },
                    "node_labels": ["Concept"],
                }
            ]
        }
    )
    client = GraphClient(neo4j=neo4j)
    node = client.find_concept_by_cui("C0025598")
    assert node is not None
    assert node.cui == "C0025598"
    assert node.preferred_name == "Metformin"
    assert "Pharmacologic Substance" in node.semantic_types
