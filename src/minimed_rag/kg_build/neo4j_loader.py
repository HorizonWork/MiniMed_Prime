"""Neo4j bulk loader for the Phase 4 KG MVP.

Schema: reified ``:Assertion`` nodes carrying per-edge predicate, confidence,
polarity, and provenance. Each assertion connects to its subject and object
via ``[:SUBJECT]`` / ``[:OBJECT]`` typed edges. PrimeKG assertions link
``:Entity``→``:Entity``; UMLS MRREL assertions link ``:Concept``→``:Concept``.
``GraphClient`` queries both via a 2-hop pattern through the Assertion node.

All loaders use ``UNWIND $rows AS row`` + ``MERGE`` so the pipeline is
idempotent and safe to re-run. The loader accepts streaming iterables so
multi-gigabyte inputs (UMLS MRREL is ~50M rows) never materialise in memory.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from minimed_rag.indexing.graph_index.neo4j_constraints import apply_constraints
from minimed_rag.indexing.graph_index.neo4j_indexes import apply_indexes


def _batched(iterable: Iterable[dict], size: int) -> Iterator[list[dict]]:
    batch: list[dict] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


CYPHER_UPSERT_ENTITY = """
UNWIND $rows AS row
MERGE (e:Entity {entity_id: row.entity_id})
SET e.entity_type = row.entity_type,
    e.preferred_name = row.preferred_name,
    e.canonical_cui = coalesce(row.canonical_cui, e.canonical_cui),
    e.source_system = row.source_system,
    e.source_id = row.source_id,
    e.graph_version = row.graph_version,
    e.is_active = coalesce(row.is_active, true)
""".strip()


CYPHER_UPSERT_CONCEPT = """
UNWIND $rows AS row
MERGE (c:Concept {cui: row.cui})
ON CREATE SET c.preferred_name = row.preferred_name,
              c.graph_version = row.graph_version,
              c.semantic_types = [],
              c.tuis = []
ON MATCH SET c.preferred_name = coalesce(row.preferred_name, c.preferred_name),
             c.graph_version = coalesce(c.graph_version, row.graph_version)
""".strip()


CYPHER_UPSERT_TERM = """
UNWIND $rows AS row
MERGE (t:Term {term_id: row.term_id})
SET t.text = row.text,
    t.normalized_text = row.normalized_text,
    t.language = row.language,
    t.source_system = row.source_system,
    t.is_preferred = coalesce(row.is_preferred, false)
WITH t, row
MATCH (c:Concept {cui: row.cui})
MERGE (t)-[:OF_CONCEPT]->(c)
""".strip()


CYPHER_SET_CONCEPT_SEMANTIC_TYPES = """
UNWIND $rows AS row
MATCH (c:Concept {cui: row.cui})
SET c.semantic_types = CASE
    WHEN c.semantic_types IS NULL THEN [row.sty]
    WHEN row.sty IN c.semantic_types THEN c.semantic_types
    ELSE c.semantic_types + row.sty
END,
    c.tuis = CASE
    WHEN c.tuis IS NULL THEN [row.tui]
    WHEN row.tui IN c.tuis THEN c.tuis
    ELSE c.tuis + row.tui
END
""".strip()


CYPHER_UPSERT_ASSERTION_ENTITY_ENDPOINTS = """
UNWIND $rows AS row
MERGE (a:Assertion {assertion_id: row.assertion_id})
SET a.predicate = row.predicate,
    a.confidence = coalesce(row.confidence, 0.5),
    a.polarity = coalesce(row.polarity, 'positive'),
    a.source_system = row.source_system,
    a.source_predicate = row.source_predicate,
    a.source_record_id = row.source_record_id,
    a.source_release = row.source_release,
    a.graph_version = row.graph_version,
    a.is_current = coalesce(row.is_current, true)
WITH a, row
MATCH (s:Entity {entity_id: row.subject_id})
MATCH (o:Entity {entity_id: row.object_id})
MERGE (a)-[:SUBJECT]->(s)
MERGE (a)-[:OBJECT]->(o)
""".strip()


CYPHER_UPSERT_ASSERTION_CONCEPT_ENDPOINTS = """
UNWIND $rows AS row
MERGE (a:Assertion {assertion_id: row.assertion_id})
SET a.predicate = row.predicate,
    a.confidence = coalesce(row.confidence, 0.5),
    a.polarity = coalesce(row.polarity, 'positive'),
    a.source_system = row.source_system,
    a.source_predicate = row.source_predicate,
    a.source_record_id = row.source_record_id,
    a.source_release = row.source_release,
    a.graph_version = row.graph_version,
    a.is_current = coalesce(row.is_current, true)
WITH a, row
MATCH (s:Concept {cui: row.subject_cui})
MATCH (o:Concept {cui: row.object_cui})
MERGE (a)-[:SUBJECT]->(s)
MERGE (a)-[:OBJECT]->(o)
""".strip()


CYPHER_LINK_ENTITY_TO_CONCEPT = """
UNWIND $rows AS row
MATCH (e:Entity {entity_id: row.entity_id})
MATCH (c:Concept {cui: row.cui})
MERGE (e)-[r:HAS_CONCEPT]->(c)
SET r.match = row.match,
    e.canonical_cui = row.cui
""".strip()


@dataclass(slots=True)
class LoaderCounts:
    entities: int = 0
    concepts: int = 0
    terms: int = 0
    semantic_types: int = 0
    assertions_entity: int = 0
    assertions_concept: int = 0
    entity_concept_links: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "entities": self.entities,
            "concepts": self.concepts,
            "terms": self.terms,
            "semantic_types": self.semantic_types,
            "assertions_entity": self.assertions_entity,
            "assertions_concept": self.assertions_concept,
            "entity_concept_links": self.entity_concept_links,
        }


@dataclass
class Neo4jLoader:
    """Stream-friendly loader for the Phase 4 reified-Assertion schema."""

    neo4j: Any
    graph_version: str = "kg_local"
    batch_size: int = 5000
    counts: LoaderCounts = field(default_factory=LoaderCounts)

    def apply_schema(self) -> None:
        apply_constraints(self.neo4j)
        apply_indexes(self.neo4j)

    def upsert_entities(self, rows: Iterable[dict]) -> int:
        return self._run_batched(CYPHER_UPSERT_ENTITY, rows, "entities")

    def upsert_concepts(self, rows: Iterable[dict]) -> int:
        return self._run_batched(CYPHER_UPSERT_CONCEPT, rows, "concepts")

    def upsert_terms(self, rows: Iterable[dict]) -> int:
        return self._run_batched(CYPHER_UPSERT_TERM, rows, "terms")

    def upsert_semantic_types(self, rows: Iterable[dict]) -> int:
        return self._run_batched(CYPHER_SET_CONCEPT_SEMANTIC_TYPES, rows, "semantic_types")

    def upsert_primekg_assertions(self, rows: Iterable[dict]) -> int:
        return self._run_batched(
            CYPHER_UPSERT_ASSERTION_ENTITY_ENDPOINTS, rows, "assertions_entity"
        )

    def upsert_umls_assertions(self, rows: Iterable[dict]) -> int:
        return self._run_batched(
            CYPHER_UPSERT_ASSERTION_CONCEPT_ENDPOINTS, rows, "assertions_concept"
        )

    def link_entity_to_concept(self, rows: Iterable[dict]) -> int:
        return self._run_batched(CYPHER_LINK_ENTITY_TO_CONCEPT, rows, "entity_concept_links")

    def counts_snapshot(self) -> dict[str, int]:
        return self.counts.as_dict()

    def load_from_projector(self, graph_projector) -> None:
        """Delegate to the canonical-KG projector (Phase 5 code path)."""
        graph_projector.build_projection(self.graph_version)

    def _run_batched(self, cypher: str, rows: Iterable[dict], counter: str) -> int:
        total = 0
        for batch in _batched(rows, self.batch_size):
            self.neo4j.run(cypher, rows=batch)
            total += len(batch)
            setattr(self.counts, counter, getattr(self.counts, counter) + len(batch))
        return total
