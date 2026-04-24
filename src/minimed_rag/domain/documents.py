"""Pseudocode domain objects: documents, sections, chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(slots=True)
class Document:
    doc_id: str
    source: str
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    publication_date: date | None = None
    license: str | None = None
    text_hash: str | None = None
    is_current: bool = True


@dataclass(slots=True)
class DocumentSection:
    doc_id: str
    section_id: str
    section_title: str
    text: str


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    doc_id: str
    text: str
    section_id: str | None = None
    section: str | None = None
    text_hash: str | None = None
    entity_ids: list[str] = field(default_factory=list)
    cuis: list[str] = field(default_factory=list)
    semantic_types: list[str] = field(default_factory=list)
    is_current: bool = True
