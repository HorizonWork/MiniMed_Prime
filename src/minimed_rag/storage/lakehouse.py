"""Pseudocode lakehouse adapter for Iceberg, Delta, or local tables."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(slots=True)
class Lakehouse:
    tables: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))

    def write(self, table: str, records: Iterable, mode: str = "append") -> None:
        rows = [record if isinstance(record, dict) else record.__dict__ for record in records]
        if mode == "overwrite":
            self.tables[table] = rows
        elif mode == "append":
            self.tables[table].extend(rows)
        else:
            raise ValueError(f"unsupported lakehouse write mode: {mode}")

    def read(self, table: str, filters: dict | None = None):
        rows = list(self.tables.get(table, []))
        if not filters:
            return rows
        return [row for row in rows if all(row.get(key) == value for key, value in filters.items())]

    def replace_partition(self, table: str, partition_filter: dict, records: Iterable) -> None:
        kept = [
            row
            for row in self.tables.get(table, [])
            if not all(row.get(k) == v for k, v in partition_filter.items())
        ]
        self.tables[table] = kept
        self.write(table, records, mode="append")
