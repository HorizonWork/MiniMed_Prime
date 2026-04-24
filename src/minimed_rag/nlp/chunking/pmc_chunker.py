"""Pseudocode pmc chunker."""

from __future__ import annotations

from minimed_rag.common.hashing import hash_text
from minimed_rag.common.ids import stable_hash
from minimed_rag.domain.documents import DocumentChunk


class PmcChunker:
    def __init__(self, max_tokens: int = 384, overlap_tokens: int = 64):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc, sections: list | None = None, section=None) -> list[DocumentChunk]:
        selected_sections = sections or [section]
        chunks = []
        for selected in selected_sections:
            text = selected.text
            chunk_id = f"CHUNK:{stable_hash(doc.doc_id, selected.section_id, text)}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    section_id=selected.section_id,
                    section=selected.section_title,
                    text=text,
                    text_hash=hash_text(text),
                )
            )
        return chunks
