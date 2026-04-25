"""Neo4j serving adapter.

Provides a minimal ``Neo4j`` wrapper over the official driver plus a
``connect()`` factory. Kept intentionally thin — higher-level primitives live
in ``kg_build/neo4j_loader.py`` and ``kg/graph_client.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Neo4j:
    driver: Any
    database: str = "neo4j"

    def run(self, cypher: str, **params) -> list[dict]:
        with self.driver.session(database=self.database) as session:
            result = session.run(cypher, **params)
            return [record.data() for record in result]

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        return self.run(cypher, **(params or {}))

    def write_batch(self, cypher: str, rows: list[dict]) -> None:
        if rows:
            self.run(cypher, rows=rows)

    def close(self) -> None:
        if self.driver is not None:
            self.driver.close()


def connect(
    uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
) -> Neo4j:
    """Construct a Neo4j adapter with a live driver. Lazy-imports the driver."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    return Neo4j(driver=driver, database=database)
