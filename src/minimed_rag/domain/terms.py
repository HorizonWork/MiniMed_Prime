"""Pseudocode domain object: term/synonym."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Term:
    term_id: str
    concept_id: str
    text: str
    normalized_text: str
    language: str = "ENG"
    source_system: str | None = None
    is_preferred: bool = False
