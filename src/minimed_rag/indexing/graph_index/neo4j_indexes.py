"""Pseudocode Neo4j indexes."""

from __future__ import annotations

INDEXES = [
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
    "CREATE INDEX assertion_predicate IF NOT EXISTS FOR (a:Assertion) ON (a.predicate)",
]


def apply_indexes(neo4j) -> None:
    for statement in INDEXES:
        neo4j.run(statement)
