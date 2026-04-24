"""Pseudocode Neo4j serving adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Neo4j:
    driver: object

    def run(self, cypher: str, **params):
        with self.driver.session() as session:
            return list(session.run(cypher, **params))

    def query(self, cypher: str, params: dict | None = None):
        return self.run(cypher, **(params or {}))

    def write_batch(self, cypher: str, rows: list[dict]) -> None:
        if rows:
            self.run(cypher, rows=rows)
