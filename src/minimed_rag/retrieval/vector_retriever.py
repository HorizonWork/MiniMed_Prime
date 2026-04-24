from __future__ import annotations

from pathlib import Path


class VectorRetriever:
    def __init__(self, milvus, query_embedder, collection: str = "minimed_chunk_dense_v1"):
        self.milvus = milvus
        self.query_embedder = query_embedder
        self.collection = collection

    def retrieve(self, query: str, filters: dict, top_k: int = 100):
        vector = self.query_embedder.encode([query])[0]
        return self.milvus.search(self.collection, vector, top_k, filters)


class LocalDenseRetriever:
    """Dense retriever backed by a local FaissIndex artifact directory."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        model_name: str | None = None,
        top_k: int = 10,
        batch_size: int = 32,
    ) -> None:
        self._index_path = Path(index_path)
        self._model_name = model_name
        self._batch_size = batch_size
        self.top_k = top_k
        self._index = None
        self._embedder = None

    def _get_index(self):
        if self._index is None:
            from minimed_rag.index.faiss_index import FaissIndex

            self._index = FaissIndex.load(self._index_path)
        return self._index

    def _get_embedder(self):
        if self._embedder is None:
            from minimed_rag.index.embedding import DEFAULT_EMBEDDING_MODEL, TransformerEmbedder

            model_name = self._model_name or self._get_index().metadata.get(
                "model_name", DEFAULT_EMBEDDING_MODEL
            )
            self._embedder = TransformerEmbedder(model_name=model_name, batch_size=self._batch_size)
        return self._embedder

    def retrieve(self, query: str, k: int | None = None):
        from minimed_rag.retrieval.bm25_retriever import RetrievalResult

        k = k if k is not None else self.top_k
        embedder = self._get_embedder()
        encode_query = getattr(embedder, "encode_queries", None) or embedder.encode
        vector = encode_query([query])[0]
        hits = self._get_index().search_with_scores(vector, k=k)
        return [RetrievalResult(chunk=chunk, score=score) for chunk, score in hits]

    @property
    def index_size(self) -> int:
        return self._get_index().index_size
