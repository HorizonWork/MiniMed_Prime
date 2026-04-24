from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minimed_rag.corpus.base import Chunk


class BM25Index:
    """Standalone BM25 index backed by rank_bm25.

    Usage:
        idx = BM25Index()
        idx.build(chunks)
        idx.save("artifacts/indexes/bm25.pkl")

        idx2 = BM25Index.load("artifacts/indexes/bm25.pkl")
        results = idx2.search("metformin diabetes", k=10)
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._chunks: list = []

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi  # type: ignore

        self._chunks = list(chunks)
        tokenized = [self._tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 10) -> list[tuple[Chunk, float]]:
        """Return up to *k* (chunk, score) pairs ranked by BM25 score."""
        if self._bm25 is None or not self._chunks:
            return []
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._chunks[i], float(scores[i])) for i in top_k if scores[i] > 0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"chunks": self._chunks, "bm25": self._bm25}, f, protocol=4)

    @classmethod
    def load(cls, path: str | Path) -> BM25Index:
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        idx = cls()
        idx._chunks = data["chunks"]
        idx._bm25 = data["bm25"]
        return idx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def __len__(self) -> int:
        return len(self._chunks)
