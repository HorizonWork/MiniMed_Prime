"""Pseudocode vector retriever."""
from __future__ import annotations


class VectorRetriever:
    def __init__(self, milvus, query_embedder, collection: str = "biomed_chunk_dense_v1"):
        self.milvus = milvus
        self.query_embedder = query_embedder
        self.collection = collection

    def retrieve(self, query: str, filters: dict, top_k: int = 100):
        vector = self.query_embedder.encode([query])[0]
        return self.milvus.search(self.collection, vector, top_k, filters)
