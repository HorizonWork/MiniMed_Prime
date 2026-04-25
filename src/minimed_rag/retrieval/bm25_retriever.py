from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minimed_rag.corpus.base import Chunk
    from minimed_rag.index.bm25_index import BM25Index


# ---------------------------------------------------------------------------
# OpenSearch-backed retriever (original, used by HybridRetriever)
# ---------------------------------------------------------------------------


class BM25Retriever:
    def __init__(self, opensearch, index: str = "minimed_chunks_v1"):
        self.opensearch = opensearch
        self.index = index

    def retrieve(self, query: str, filters: dict, top_k: int = 100):
        body = {
            "query": {
                "bool": {
                    "must": [{"match": {"text": query}}],
                    "filter": [{"term": {key: value}} for key, value in filters.items()],
                }
            },
            "size": top_k,
        }
        return self.opensearch.search(self.index, body)


# ---------------------------------------------------------------------------
# Standalone retriever — no external services required
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


class LocalBM25Retriever:
    """BM25 retriever backed by a pre-built BM25Index pickle.

    Works fully offline — no OpenSearch/Milvus needed.
    """

    def __init__(self, index_path: str | Path, top_k: int = 10) -> None:
        self._index_path = Path(index_path)
        self.top_k = top_k
        self._index: BM25Index | None = None

    # Lazy-load so the CLI can import without touching disk at import time
    def _get_index(self) -> BM25Index:
        if self._index is None:
            from minimed_rag.index.bm25_index import BM25Index

            self._index = BM25Index.load(self._index_path)
        return self._index

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        k = k if k is not None else self.top_k
        hits = self._get_index().search(query, k=k)
        return [RetrievalResult(chunk=chunk, score=score) for chunk, score in hits]

    @property
    def index_size(self) -> int:
        return len(self._get_index())
