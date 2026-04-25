"""Neo4j indexes for the Phase 4 KG schema."""

from __future__ import annotations

INDEXES = [
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
    "CREATE INDEX entity_cui IF NOT EXISTS FOR (e:Entity) ON (e.canonical_cui)",
    "CREATE INDEX entity_source IF NOT EXISTS FOR (e:Entity) ON (e.source_system, e.source_id)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.preferred_name)",
    "CREATE INDEX term_normalized IF NOT EXISTS FOR (t:Term) ON (t.normalized_text)",
    "CREATE INDEX assertion_predicate IF NOT EXISTS FOR (a:Assertion) ON (a.predicate)",
    "CREATE INDEX assertion_current IF NOT EXISTS FOR (a:Assertion) ON (a.is_current)",
    "CREATE INDEX assertion_source IF NOT EXISTS FOR (a:Assertion) ON (a.source_system)",
]

FULLTEXT_INDEXES = [
    "CREATE FULLTEXT INDEX entity_name_ft IF NOT EXISTS FOR (e:Entity) ON EACH [e.preferred_name]",
    "CREATE FULLTEXT INDEX term_text_ft IF NOT EXISTS FOR (t:Term) ON EACH [t.text, t.normalized_text]",
]


def apply_indexes(neo4j) -> None:
    for statement in INDEXES + FULLTEXT_INDEXES:
        neo4j.run(statement)
