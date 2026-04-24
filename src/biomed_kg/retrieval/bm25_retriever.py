"""Pseudocode BM25 retriever."""
from __future__ import annotations


class BM25Retriever:
    def __init__(self, opensearch, index: str = "biomed_chunks_v1"):
        self.opensearch = opensearch
        self.index = index

    def retrieve(self, query: str, filters: dict, top_k: int = 100):
        body = {"query": {"bool": {"must": [{"match": {"text": query}}], "filter": [{"term": {key: value}} for key, value in filters.items()]}}, "size": top_k}
        return self.opensearch.search(self.index, body)
