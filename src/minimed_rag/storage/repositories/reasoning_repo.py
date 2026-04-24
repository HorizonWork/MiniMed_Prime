"""Pseudocode repository for reasoning_repo."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(slots=True)
class ReasoningRepo:
    rows: dict[str, object] = field(default_factory=dict)

    def get(self, key: str):
        return self.rows.get(key)

    def upsert(self, key: str, value: object) -> None:
        self.rows[key] = value

    def write_many(self, records: Iterable, key_attr: str = "id") -> None:
        for record in records:
            key = getattr(record, key_attr, None)
            if key is None and isinstance(record, dict):
                key = record.get(key_attr)
            self.rows[str(key)] = record

    def iter_all(self):
        return iter(self.rows.values())
