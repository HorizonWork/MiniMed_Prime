"""Pseudocode domain object: evidence span."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(slots=True)
class EvidenceSpan:
    evidence_id: str
    doc_id: str
    chunk_id: str
    text: str
    section: str | None = None
    offset_start: int | None = None
    offset_end: int | None = None
    evidence_type: str = "text"
    publication_date: date | None = None
    source_system: str = ""
