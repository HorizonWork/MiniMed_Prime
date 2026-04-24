"""Pseudocode graph index pipeline."""

from __future__ import annotations

from minimed_rag.indexing.graph_index.neo4j_constraints import apply_constraints
from minimed_rag.indexing.graph_index.neo4j_indexes import apply_indexes


class GraphIndexPipeline:
    def __init__(self, neo4j):
        self.neo4j = neo4j

    def run(self, graph_version: str) -> None:
        apply_constraints(self.neo4j)
        apply_indexes(self.neo4j)
