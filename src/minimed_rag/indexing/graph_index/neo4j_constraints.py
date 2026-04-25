"""Neo4j constraints for the Phase 4 KG schema.

Four node labels with primary-key uniqueness:

- ``:Entity``    — canonical KG node (PrimeKG / future canonical builder)
- ``:Concept``   — UMLS concept (CUI-native)
- ``:Term``      — UMLS term / synonym
- ``:Assertion`` — reified edge carrying predicate, confidence, provenance
"""

from __future__ import annotations

CONSTRAINTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT concept_cui IF NOT EXISTS FOR (c:Concept) REQUIRE c.cui IS UNIQUE",
    "CREATE CONSTRAINT term_id IF NOT EXISTS FOR (t:Term) REQUIRE t.term_id IS UNIQUE",
    "CREATE CONSTRAINT assertion_id IF NOT EXISTS FOR (a:Assertion) REQUIRE a.assertion_id IS UNIQUE",
]


def apply_constraints(neo4j) -> None:
    for statement in CONSTRAINTS:
        neo4j.run(statement)
