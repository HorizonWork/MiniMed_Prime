from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from minimed_rag.retrieval.bm25_retriever import RetrievalResult


@dataclass(slots=True)
class HybridHit:
    result: RetrievalResult
    rrf_score: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    """Hybrid local retriever that fuses BM25 and dense results with RRF."""

    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        *,
        top_k: int = 10,
        bm25_k: int = 50,
        dense_k: int = 50,
        rrf_k: int = 60,
        weights: dict[str, float] | None = None,
        reranker=None,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.top_k = top_k
        self.bm25_k = bm25_k
        self.dense_k = dense_k
        self.rrf_k = rrf_k
        self.weights = weights or {"bm25": 1.0, "dense": 1.0}
        self.reranker = reranker

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        limit = k if k is not None else self.top_k
        with ThreadPoolExecutor(max_workers=2) as pool:
            bm25_future = pool.submit(self.bm25_retriever.retrieve, query, self.bm25_k)
            dense_future = pool.submit(self.dense_retriever.retrieve, query, self.dense_k)
            bm25_results = bm25_future.result()
            dense_results = dense_future.result()

        fused = reciprocal_rank_fusion(
            {"bm25": bm25_results, "dense": dense_results},
            rrf_k=self.rrf_k,
            weights=self.weights,
        )
        results = [RetrievalResult(chunk=hit.result.chunk, score=hit.rrf_score) for hit in fused]
        if self.reranker is not None:
            results = self.reranker.rerank(query=query, results=results)
        return results[:limit]


def reciprocal_rank_fusion(
    ranked_lists: dict[str, Iterable[RetrievalResult]],
    *,
    rrf_k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[HybridHit]:
    weights = weights or {}
    merged: dict[str, HybridHit] = {}

    for source, results in ranked_lists.items():
        weight = weights.get(source, 1.0)
        for rank, result in enumerate(results, start=1):
            key = _chunk_key(result)
            hit = merged.get(key)
            if hit is None:
                hit = HybridHit(result=result)
                merged[key] = hit
            hit.rrf_score += weight / (rrf_k + rank)
            hit.ranks[source] = rank
            hit.scores[source] = result.score

    return sorted(merged.values(), key=lambda hit: hit.rrf_score, reverse=True)


def _chunk_key(result: RetrievalResult) -> str:
    chunk = result.chunk
    source = getattr(chunk, "source", "")
    chunk_id = getattr(chunk, "id", "")
    return f"{source}:{chunk_id}"


__all__ = ["HybridHit", "HybridRetriever", "reciprocal_rank_fusion"]
