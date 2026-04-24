from __future__ import annotations

import numpy as np
import pytest

from minimed_rag.corpus.base import Chunk
from minimed_rag.index.embedding import TransformerEmbedder
from minimed_rag.index.faiss_index import FaissIndex
from minimed_rag.retrieval.bm25_retriever import RetrievalResult
from minimed_rag.retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from minimed_rag.retrieval.vector_retriever import LocalDenseRetriever


def test_transformer_embedder_empty_batch_does_not_load_model():
    embedder = TransformerEmbedder()
    assert embedder.encode([]).shape == (0, 0)


def test_faiss_index_round_trip(tmp_path):
    chunks = [
        Chunk(id="c1", text="aspirin inhibits platelet aggregation", source="unit"),
        Chunk(id="c2", text="insulin lowers blood glucose", source="unit"),
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    index_path = tmp_path / "dense"
    index = FaissIndex(path=index_path)
    index.build(chunks, embeddings)
    index.save()

    loaded = FaissIndex.load(index_path)
    hits = loaded.search_with_scores(np.array([1.0, 0.0], dtype=np.float32), k=1)

    assert loaded.index_size == 2
    assert hits[0][0].id == "c1"
    assert hits[0][1] == pytest.approx(1.0)


def test_reciprocal_rank_fusion_rewards_overlap():
    a = RetrievalResult(Chunk(id="a", text="alpha", source="unit"), 9.0)
    b = RetrievalResult(Chunk(id="b", text="beta", source="unit"), 8.0)
    c = RetrievalResult(Chunk(id="c", text="gamma", source="unit"), 7.0)

    fused = reciprocal_rank_fusion({"bm25": [a, b], "dense": [c, a]}, rrf_k=60)

    assert fused[0].result.chunk.id == "a"
    assert fused[0].ranks == {"bm25": 1, "dense": 2}


def test_hybrid_retriever_returns_rrf_results():
    a = RetrievalResult(Chunk(id="a", text="alpha", source="unit"), 9.0)
    b = RetrievalResult(Chunk(id="b", text="beta", source="unit"), 8.0)
    c = RetrievalResult(Chunk(id="c", text="gamma", source="unit"), 7.0)

    hybrid = HybridRetriever(
        bm25_retriever=_StaticRetriever([a, b]),
        dense_retriever=_StaticRetriever([c, a]),
        top_k=2,
        bm25_k=2,
        dense_k=2,
    )

    results = hybrid.retrieve("alpha")

    assert [result.chunk.id for result in results] == ["a", "c"]


def test_local_dense_retriever_uses_query_encoder():
    retriever = LocalDenseRetriever(index_path="/tmp/not-used", top_k=1)
    retriever._embedder = _FakeEmbedder()
    retriever._index = _FakeIndex()

    results = retriever.retrieve("aspirin")

    assert len(results) == 1
    assert results[0].chunk.id == "dense-hit"
    assert results[0].score == pytest.approx(0.75)


class _StaticRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        return self.results[:k]


class _FakeEmbedder:
    def encode_queries(self, texts: list[str]) -> np.ndarray:
        assert texts == ["aspirin"]
        return np.array([[1.0, 0.0]], dtype=np.float32)


class _FakeIndex:
    metadata = {"model_name": "fake"}
    index_size = 1

    def search_with_scores(self, query_embedding: np.ndarray, k: int = 10):
        assert query_embedding.tolist() == [1.0, 0.0]
        chunk = Chunk(id="dense-hit", text="aspirin platelet evidence", source="unit")
        return [(chunk, 0.75)]
