"""Pseudocode embedding pipeline."""

from __future__ import annotations

from minimed_rag.common.ids import make_vector_id
from minimed_rag.kg_build.graph_projector import batch_iter


class EmbeddingPipeline:
    def __init__(
        self, lakehouse, milvus, concept_embedder=None, chunk_embedder=None, assertion_embedder=None
    ):
        self.lakehouse = lakehouse
        self.milvus = milvus
        self.concept_embedder = concept_embedder
        self.chunk_embedder = chunk_embedder
        self.assertion_embedder = assertion_embedder

    def run(self, graph_version: str) -> None:
        self.embed_concepts(graph_version)
        self.embed_chunks(graph_version)
        self.embed_assertions(graph_version)

    def embed_concepts(self, graph_version: str) -> None:
        return None

    def embed_assertions(self, graph_version: str) -> None:
        return None

    def embed_chunks(self, graph_version: str) -> None:
        chunks = self.lakehouse.read(
            "document_chunk", {"graph_version": graph_version, "is_current": True}
        )
        for batch in batch_iter(chunks, 128):
            texts = [self.prepare_chunk_text(chunk) for chunk in batch]
            vectors = self.chunk_embedder.encode(texts)
            records = []
            for chunk, vector in zip(batch, vectors):
                records.append(
                    {
                        "vector_id": make_vector_id("chunk", chunk["chunk_id"]),
                        "object_type": "chunk",
                        "object_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "source": chunk.get("source"),
                        "section": chunk.get("section"),
                        "entity_ids": chunk.get("entity_ids", []),
                        "cuis": chunk.get("cuis", []),
                        "embedding_model": self.chunk_embedder.model_name,
                        "embedding_dim": len(vector),
                        "text_hash": chunk.get("text_hash"),
                        "vector": vector,
                        "graph_version": graph_version,
                    }
                )
            self.milvus.insert(collection="minimed_chunk_dense_v1", records=records)

    def prepare_chunk_text(self, chunk: dict) -> str:
        return self.chunk_embedder.prepare_chunk_text(chunk)
