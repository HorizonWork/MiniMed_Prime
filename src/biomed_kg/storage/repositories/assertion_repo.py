"""Pseudocode repository for canonical assertions."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AssertionRepo:
    rows: dict[str, object] = field(default_factory=dict)

    def upsert(self, assertion) -> None:
        self.rows[assertion.assertion_id] = assertion

    def get(self, assertion_id: str):
        return self.rows.get(assertion_id)

    def iter_current_assertions(self):
        return (a for a in self.rows.values() if getattr(a, "is_current", False))

    def update(self, assertion) -> None:
        self.rows[assertion.assertion_id] = assertion
