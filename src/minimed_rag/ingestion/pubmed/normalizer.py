"""Pseudocode PubMed document normalizer."""

from __future__ import annotations

from minimed_rag.common.hashing import hash_text
from minimed_rag.domain.documents import Document, DocumentSection


def normalize_pubmed_document(citation) -> Document:
    pmid = citation.payload.get("pmid", "UNKNOWN")
    title = citation.payload.get("title", "")
    text = citation.payload.get("abstract", "")
    return Document(doc_id=f"PMID:{pmid}", source="pubmed", title=title, text_hash=hash_text(text))


def extract_pubmed_sections(citation) -> list[DocumentSection]:
    doc_id = f"PMID:{citation.payload.get('pmid', 'UNKNOWN')}"
    return [
        DocumentSection(
            doc_id=doc_id,
            section_id=f"{doc_id}:abstract",
            section_title="Abstract",
            text=citation.payload.get("abstract", ""),
        )
    ]
