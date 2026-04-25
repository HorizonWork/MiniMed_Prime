"""Postgres-backed ConceptRepository for the Phase 4 entity linker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ConceptRow:
    concept_id: str
    cui: str
    preferred_name: str
    graph_version: str

    @classmethod
    def from_row(cls, row: tuple[Any, ...]) -> ConceptRow:
        return cls(
            concept_id=row[0],
            cui=row[1],
            preferred_name=row[2],
            graph_version=row[3],
        )


@dataclass(slots=True)
class ConceptRepository:
    postgres: Any

    def get_by_cui(self, cui: str) -> ConceptRow | None:
        rows = self.postgres.execute(
            "SELECT concept_id, cui, preferred_name, graph_version "
            "FROM concept WHERE cui = %(cui)s",
            {"cui": cui},
        )
        if not rows:
            return None
        return ConceptRow.from_row(rows[0])

    def get_by_sab_code(self, sab: str, code: str) -> list[str]:
        """Return CUIs that match a (SAB, CODE) pair (UMLS crosswalk)."""
        rows = self.postgres.execute(
            "SELECT cui FROM umls_crosswalk WHERE sab = %(sab)s AND code = %(code)s",
            {"sab": sab, "code": code},
        )
        return [row[0] for row in (rows or [])]

    def count(self) -> int:
        rows = self.postgres.execute("SELECT COUNT(*) FROM concept")
        return int(rows[0][0]) if rows else 0
