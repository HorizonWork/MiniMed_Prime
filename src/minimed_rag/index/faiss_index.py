from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from minimed_rag.corpus.base import Chunk


def _load_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except ImportError:
        return None


@dataclass(slots=True)
class FaissIndex:
    """Local dense index with FAISS when available and numpy cosine fallback."""

    path: str | Path = "artifacts/indexes/dense_faiss"
    normalize: bool = True
    metadata: dict = field(default_factory=dict)
    _chunks: list[Chunk] = field(default_factory=list, init=False, repr=False)
    _vectors: np.ndarray | None = field(default=None, init=False, repr=False)
    _index: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"embeddings must be a 2D array, got shape={matrix.shape}")
        if len(chunks) != matrix.shape[0]:
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} != {matrix.shape[0]}"
            )

        if self.normalize:
            matrix = self._l2_normalize(matrix)

        self._chunks = list(chunks)
        self._vectors = matrix
        self._index = self._build_faiss_index(matrix)

    def save(self, path: str | Path | None = None) -> None:
        target = Path(path) if path is not None else Path(self.path)
        target.mkdir(parents=True, exist_ok=True)

        with (target / "chunks.pkl").open("wb") as f:
            pickle.dump(self._chunks, f, protocol=4)
        if self._vectors is None:
            raise ValueError("index has not been built")
        np.save(target / "vectors.npy", self._vectors)

        faiss = _load_faiss()
        if faiss is not None and self._index is not None:
            faiss.write_index(self._index, str(target / "index.faiss"))

        metadata = {
            **self.metadata,
            "num_chunks": len(self._chunks),
            "embedding_dim": int(self._vectors.shape[1]),
            "normalize": self.normalize,
            "backend": "faiss" if faiss is not None and self._index is not None else "numpy",
        }
        (target / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> FaissIndex:
        target = Path(path)
        metadata_path = target / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        idx = cls(path=target, normalize=bool(metadata.get("normalize", True)), metadata=metadata)

        with (target / "chunks.pkl").open("rb") as f:
            idx._chunks = pickle.load(f)
        idx._vectors = np.load(target / "vectors.npy").astype(np.float32, copy=False)

        faiss = _load_faiss()
        faiss_path = target / "index.faiss"
        if faiss is not None and faiss_path.exists():
            idx._index = faiss.read_index(str(faiss_path))
        else:
            idx._index = idx._build_faiss_index(idx._vectors)
        return idx

    def search(self, query_embedding: np.ndarray, k: int = 10) -> list[Chunk]:
        return [chunk for chunk, _score in self.search_with_scores(query_embedding, k=k)]

    def search_with_scores(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> list[tuple[Chunk, float]]:
        if not self._chunks or self._vectors is None:
            return []

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 2:
            if query.shape[0] != 1:
                raise ValueError("query_embedding must be a single vector")
            query = query[0]
        if query.ndim != 1:
            raise ValueError(f"query_embedding must be 1D or shape=(1, dim), got {query.shape}")
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"query dim {query.shape[0]} does not match index dim {self._vectors.shape[1]}"
            )

        if self.normalize:
            query = self._l2_normalize(query.reshape(1, -1))[0]

        limit = min(k, len(self._chunks))
        if limit <= 0:
            return []

        if self._index is not None:
            scores, indexes = self._index.search(query.reshape(1, -1), limit)
            pairs = zip(indexes[0].tolist(), scores[0].tolist(), strict=False)
            return [
                (self._chunks[i], float(score))
                for i, score in pairs
                if i >= 0 and i < len(self._chunks)
            ]

        scores = self._vectors @ query
        top = np.argsort(scores)[::-1][:limit]
        return [(self._chunks[int(i)], float(scores[int(i)])) for i in top]

    @property
    def index_size(self) -> int:
        return len(self._chunks)

    def _build_faiss_index(self, matrix: np.ndarray):
        faiss = _load_faiss()
        if faiss is None:
            return None
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return index

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


__all__ = ["FaissIndex"]
