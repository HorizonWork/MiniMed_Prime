"""Pseudocode Neo4j constraints."""
from __future__ import annotations


CONSTRAINTS = [
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE CONSTRAINT assertion_id IF NOT EXISTS FOR (a:Assertion) REQUIRE a.assertion_id IS UNIQUE",
]


def apply_constraints(neo4j) -> None:
    for statement in CONSTRAINTS:
        neo4j.run(statement)
