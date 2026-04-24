"""Pseudocode Neo4j loader facade."""

from __future__ import annotations


class Neo4jLoader:
    def __init__(self, graph_projector):
        self.graph_projector = graph_projector

    def load(self, graph_version: str) -> None:
        self.graph_projector.build_projection(graph_version)
