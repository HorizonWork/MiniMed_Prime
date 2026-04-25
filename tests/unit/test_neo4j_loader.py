"""Unit tests for the Neo4j loader.

Uses a recording mock for the Neo4j adapter so the tests run without a live
driver. Assertions are on the Cypher string shape and the batched params.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from minimed_rag.kg_build.neo4j_loader import (
    CYPHER_LINK_ENTITY_TO_CONCEPT,
    CYPHER_SET_CONCEPT_SEMANTIC_TYPES,
    CYPHER_UPSERT_ASSERTION_CONCEPT_ENDPOINTS,
    CYPHER_UPSERT_ASSERTION_ENTITY_ENDPOINTS,
    CYPHER_UPSERT_CONCEPT,
    CYPHER_UPSERT_ENTITY,
    CYPHER_UPSERT_TERM,
    Neo4jLoader,
)


@dataclass
class RecordingNeo4j:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def run(self, cypher: str, **params: Any) -> list:
        self.calls.append((cypher, params))
        return []


@pytest.fixture
def neo4j() -> RecordingNeo4j:
    return RecordingNeo4j()


@pytest.fixture
def loader(neo4j: RecordingNeo4j) -> Neo4jLoader:
    return Neo4jLoader(neo4j=neo4j, graph_version="kg_test", batch_size=2)


def _rows(n: int) -> list[dict]:
    return [{"entity_id": f"KG:Drug:{i}", "preferred_name": f"drug_{i}"} for i in range(n)]


def test_apply_schema_runs_constraints_then_indexes(loader, neo4j):
    loader.apply_schema()
    statements = [call[0] for call in neo4j.calls]

    # Constraints come first (uniqueness for MERGE speed), then indexes.
    assert any("CREATE CONSTRAINT entity_id" in s for s in statements)
    assert any("CREATE CONSTRAINT concept_cui" in s for s in statements)
    assert any("CREATE CONSTRAINT term_id" in s for s in statements)
    assert any("CREATE CONSTRAINT assertion_id" in s for s in statements)
    assert any("CREATE INDEX entity_type" in s for s in statements)
    assert any("CREATE INDEX entity_cui" in s for s in statements)
    assert any("CREATE FULLTEXT INDEX entity_name_ft" in s for s in statements)


def test_upsert_entities_batches_input(loader, neo4j):
    rows = _rows(5)
    written = loader.upsert_entities(iter(rows))

    assert written == 5
    assert loader.counts.entities == 5

    # batch_size=2 → three calls (2, 2, 1)
    entity_calls = [c for c in neo4j.calls if CYPHER_UPSERT_ENTITY in c[0]]
    assert len(entity_calls) == 3
    assert [len(c[1]["rows"]) for c in entity_calls] == [2, 2, 1]
    assert entity_calls[0][1]["rows"][0]["entity_id"] == "KG:Drug:0"


def test_upsert_concepts_uses_cui_merge(loader, neo4j):
    rows = [{"cui": "C0025598", "preferred_name": "Metformin", "graph_version": "kg_test"}]
    loader.upsert_concepts(rows)

    calls = [c for c in neo4j.calls if CYPHER_UPSERT_CONCEPT in c[0]]
    assert calls, "expected concept MERGE call"
    assert "MERGE (c:Concept {cui: row.cui})" in calls[0][0]
    assert calls[0][1]["rows"][0]["cui"] == "C0025598"


def test_upsert_terms_links_to_concept(loader, neo4j):
    rows = [
        {
            "term_id": "AUI:1234",
            "text": "Metformin hydrochloride",
            "normalized_text": "metformin hydrochloride",
            "language": "ENG",
            "source_system": "umls",
            "is_preferred": False,
            "cui": "C0025598",
        }
    ]
    loader.upsert_terms(rows)

    calls = [c for c in neo4j.calls if CYPHER_UPSERT_TERM in c[0]]
    assert calls
    cypher = calls[0][0]
    assert "MERGE (t:Term {term_id: row.term_id})" in cypher
    assert "MERGE (t)-[:OF_CONCEPT]->(c)" in cypher


def test_upsert_primekg_assertions_entity_endpoints(loader, neo4j):
    rows = [
        {
            "assertion_id": "ASSERT:abc",
            "predicate": "indication",
            "subject_id": "KG:Drug:1",
            "object_id": "KG:Disease:1",
            "confidence": 0.9,
            "source_system": "primekg",
            "source_predicate": "indication",
            "source_record_id": "rec1",
            "source_release": "local",
            "graph_version": "kg_test",
        }
    ]
    loader.upsert_primekg_assertions(rows)

    calls = [c for c in neo4j.calls if CYPHER_UPSERT_ASSERTION_ENTITY_ENDPOINTS in c[0]]
    assert calls
    cypher = calls[0][0]
    assert "MATCH (s:Entity {entity_id: row.subject_id})" in cypher
    assert "MATCH (o:Entity {entity_id: row.object_id})" in cypher
    assert "MERGE (a)-[:SUBJECT]->(s)" in cypher
    assert "MERGE (a)-[:OBJECT]->(o)" in cypher
    assert loader.counts.assertions_entity == 1


def test_upsert_umls_assertions_concept_endpoints(loader, neo4j):
    rows = [
        {
            "assertion_id": "ASSERT:def",
            "predicate": "is_a",
            "subject_cui": "C1234567",
            "object_cui": "C7654321",
            "confidence": 0.85,
            "source_system": "umls",
            "source_predicate": "PAR",
            "source_record_id": "rec2",
            "source_release": "local",
            "graph_version": "kg_test",
        }
    ]
    loader.upsert_umls_assertions(rows)

    calls = [c for c in neo4j.calls if CYPHER_UPSERT_ASSERTION_CONCEPT_ENDPOINTS in c[0]]
    assert calls
    cypher = calls[0][0]
    assert "MATCH (s:Concept {cui: row.subject_cui})" in cypher
    assert "MATCH (o:Concept {cui: row.object_cui})" in cypher
    assert loader.counts.assertions_concept == 1


def test_upsert_semantic_types_dedupes(loader, neo4j):
    rows = [{"cui": "C0025598", "tui": "T121", "sty": "Pharmacologic Substance"}]
    loader.upsert_semantic_types(rows)

    calls = [c for c in neo4j.calls if CYPHER_SET_CONCEPT_SEMANTIC_TYPES in c[0]]
    assert calls
    cypher = calls[0][0]
    assert "WHEN row.sty IN c.semantic_types THEN c.semantic_types" in cypher


def test_link_entity_to_concept_sets_canonical_cui(loader, neo4j):
    rows = [{"entity_id": "KG:Drug:1", "cui": "C0025598", "match": "source_code_crosswalk"}]
    loader.link_entity_to_concept(rows)

    calls = [c for c in neo4j.calls if CYPHER_LINK_ENTITY_TO_CONCEPT in c[0]]
    assert calls
    cypher = calls[0][0]
    assert "MERGE (e)-[r:HAS_CONCEPT]->(c)" in cypher
    assert "e.canonical_cui = row.cui" in cypher
    assert loader.counts.entity_concept_links == 1


def test_counts_snapshot_aggregates(loader):
    loader.counts.entities = 10
    loader.counts.assertions_entity = 25
    snapshot = loader.counts_snapshot()
    assert snapshot["entities"] == 10
    assert snapshot["assertions_entity"] == 25
    assert snapshot["concepts"] == 0


def test_batch_empty_iterable_is_noop(loader, neo4j):
    written = loader.upsert_entities(iter([]))
    assert written == 0
    assert loader.counts.entities == 0
    assert neo4j.calls == []


def test_legacy_load_from_projector_delegates(loader):
    class FakeProjector:
        called_with: str | None = None

        def build_projection(self, graph_version: str) -> None:
            self.called_with = graph_version

    projector = FakeProjector()
    loader.load_from_projector(projector)
    assert projector.called_with == "kg_test"
