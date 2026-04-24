"""Pseudocode StatPearls normalizer."""
from __future__ import annotations

from biomed_kg.common.ids import make_section_id
from biomed_kg.domain.documents import Document, DocumentSection


def normalize_statpearls_chapter(chapter, statpearls_chunker):
    doc = Document(
        doc_id=f"STATPEARLS:{chapter.nbk_id}",
        source="statpearls",
        title=chapter.title,
        authors=getattr(chapter, "authors", []),
        publication_date=getattr(chapter, "updated_at", None),
        license=getattr(chapter, "license", None),
    )
    sections = []
    chunks = []
    for section in chapter.sections:
        title = section.title if hasattr(section, "title") else section.get("title", "")
        text = section.text if hasattr(section, "text") else section.get("text", "")
        normalized_section = DocumentSection(doc_id=doc.doc_id, section_id=make_section_id(doc.doc_id, title), section_title=title, text=text)
        sections.append(normalized_section)
        chunks.extend(statpearls_chunker.chunk(doc=doc, section=normalized_section))
    return [doc, *sections, *chunks]
