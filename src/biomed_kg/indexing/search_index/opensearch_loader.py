"""Pseudocode OpenSearch loader."""
from __future__ import annotations


class OpenSearchLoader:
    def __init__(self, opensearch):
        self.opensearch = opensearch

    def load_documents(self, index: str, docs: list[dict]) -> None:
        self.opensearch.bulk_index(index=index, docs=docs)
