"""Pseudocode Neo4j graph projection."""

from __future__ import annotations


def batch_iter(rows, size: int):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class Neo4jGraphProjector:
    def __init__(self, lakehouse, neo4j):
        self.lakehouse = lakehouse
        self.neo4j = neo4j

    def build_projection(self, graph_version: str) -> None:
        self.create_constraints()
        self.load_entities(graph_version)
        self.load_concepts(graph_version)
        self.load_terms(graph_version)
        self.load_assertion_nodes(graph_version)
        self.load_evidence_nodes(graph_version)
        self.load_assertion_relationships(graph_version)
        self.load_materialized_edges(graph_version)
        self.create_indexes()

    def create_constraints(self) -> None:
        self.neo4j.run(
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
        )

    def create_indexes(self) -> None:
        self.neo4j.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)")

    def load_entities(self, graph_version: str) -> None:
        query = """
        UNWIND $rows AS row
        MERGE (e:Entity {entity_id: row.entity_id})
        SET e.preferred_name = row.preferred_name,
            e.entity_type = row.entity_type,
            e.canonical_cui = row.canonical_cui,
            e.graph_version = row.graph_version
        """
        for batch in batch_iter(
            self.lakehouse.read("entity", {"graph_version": graph_version, "is_active": True}),
            10000,
        ):
            self.neo4j.run(query, rows=batch)

    def load_concepts(self, graph_version: str) -> None:
        return None

    def load_terms(self, graph_version: str) -> None:
        return None

    def load_assertion_nodes(self, graph_version: str) -> None:
        query = """
        UNWIND $rows AS row
        MERGE (a:Assertion {assertion_id: row.assertion_id})
        SET a.predicate = row.predicate,
            a.confidence = row.confidence,
            a.source_system = row.source_system,
            a.source_release = row.source_release,
            a.graph_version = row.graph_version,
            a.is_current = row.is_current
        """
        for batch in batch_iter(
            self.lakehouse.read("assertion", {"graph_version": graph_version, "is_current": True}),
            10000,
        ):
            self.neo4j.run(query, rows=batch)

    def load_evidence_nodes(self, graph_version: str) -> None:
        return None

    def load_assertion_relationships(self, graph_version: str) -> None:
        query = """
        UNWIND $rows AS row
        MATCH (a:Assertion {assertion_id: row.assertion_id})
        MATCH (s:Entity {entity_id: row.subject_id})
        MATCH (o:Entity {entity_id: row.object_id})
        MERGE (a)-[:SUBJECT]->(s)
        MERGE (a)-[:OBJECT]->(o)
        """
        for batch in batch_iter(self.lakehouse.read("assertion"), 10000):
            self.neo4j.run(query, rows=batch)

    def load_materialized_edges(self, graph_version: str) -> None:
        return None
