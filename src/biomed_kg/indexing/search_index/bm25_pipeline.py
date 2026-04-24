"""Pseudocode BM25/OpenSearch indexing pipeline."""
from __future__ import annotations

from biomed_kg.kg_build.graph_projector import batch_iter


class SearchIndexPipeline:
    def __init__(self, lakehouse, opensearch):
        self.lakehouse = lakehouse
        self.opensearch = opensearch

    def index_chunks(self, graph_version: str) -> None:
        chunks = self.lakehouse.read("document_chunk", {"graph_version": graph_version, "is_current": True})
        for batch in batch_iter(chunks, 1000):
            docs = []
            for chunk in batch:
                docs.append({"_id": chunk["chunk_id"], "doc_id": chunk["doc_id"], "source": chunk.get("source"), "title": chunk.get("document_title"), "section": chunk.get("section"), "text": chunk["text"], "entity_ids": chunk.get("entity_ids", []), "cuis": chunk.get("cuis", []), "semantic_types": chunk.get("semantic_types", []), "publication_date": chunk.get("publication_date"), "graph_version": graph_version})
            self.opensearch.bulk_index(index="biomed_chunks_v1", docs=docs)
