"""Pseudocode vector index build pipeline."""
from __future__ import annotations


class VectorIndexPipeline:
    def __init__(self, embedding_pipeline):
        self.embedding_pipeline = embedding_pipeline

    def run(self, graph_version: str) -> None:
        self.embedding_pipeline.run(graph_version)
