"""Pseudocode chunk embedder."""

from __future__ import annotations


class ChunkEmbedder:
    model_name = "chunk_embedder"

    def prepare_chunk_text(self, chunk: dict) -> str:
        return (
            f"[SOURCE] {chunk.get('source', '')}\n"
            f"[TITLE] {chunk.get('document_title', '')}\n"
            f"[SECTION] {chunk.get('section', '')}\n"
            f"[TEXT] {chunk.get('text', '')}"
        ).strip()

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]
