"""Postgres-backed TermRepository powering the entity-linker fallback tier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TermMatch:
    term_id: str
    concept_id: str
    cui: str
    text: str
    normalized_text: str
    language: str
    source_system: str
    preferred_name: str | None = None
    score: float = 1.0


@dataclass(slots=True)
class TermRepository:
    """Fast substring / exact lookup against the UMLS-loaded ``term`` table.

    Relies on the ``idx_term_normalized_text`` B-tree index created by
    ``data_contracts/sql/006_phase4_kg.sql``. Phase 4 scope only exposes
    exact-match and LIKE-prefix search; Phase 5 can layer pg_trgm /
    full-text once volume justifies it.
    """

    postgres: Any

    def search_by_normalized_text(self, normalized: str, limit: int = 10) -> list[TermMatch]:
        rows = self.postgres.execute(
            """
            SELECT t.term_id, t.concept_id, c.cui, t.text, t.normalized_text,
                   t.language, t.source_system, c.preferred_name
            FROM term t
            JOIN concept c ON c.concept_id = t.concept_id
            WHERE t.normalized_text = %(q)s
            ORDER BY (c.preferred_name = t.text) DESC, t.text
            LIMIT %(limit)s
            """,
            {"q": normalized, "limit": limit},
        )
        return [_row_to_match(row) for row in (rows or [])]

    def search_by_prefix(self, normalized_prefix: str, limit: int = 20) -> list[TermMatch]:
        rows = self.postgres.execute(
            """
            SELECT t.term_id, t.concept_id, c.cui, t.text, t.normalized_text,
                   t.language, t.source_system, c.preferred_name
            FROM term t
            JOIN concept c ON c.concept_id = t.concept_id
            WHERE t.normalized_text LIKE %(q)s
            ORDER BY length(t.normalized_text), t.text
            LIMIT %(limit)s
            """,
            {"q": normalized_prefix + "%", "limit": limit},
        )
        return [_row_to_match(row) for row in (rows or [])]

    def get_by_concept(self, concept_id: str) -> list[TermMatch]:
        rows = self.postgres.execute(
            """
            SELECT t.term_id, t.concept_id, c.cui, t.text, t.normalized_text,
                   t.language, t.source_system, c.preferred_name
            FROM term t
            JOIN concept c ON c.concept_id = t.concept_id
            WHERE t.concept_id = %(cid)s
            """,
            {"cid": concept_id},
        )
        return [_row_to_match(row) for row in (rows or [])]


def _row_to_match(row: tuple[Any, ...]) -> TermMatch:
    return TermMatch(
        term_id=row[0],
        concept_id=row[1],
        cui=row[2],
        text=row[3],
        normalized_text=row[4],
        language=row[5],
        source_system=row[6],
        preferred_name=row[7],
    )
