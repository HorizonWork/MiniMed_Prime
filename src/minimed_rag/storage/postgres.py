"""Pseudocode PostgreSQL metadata adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Postgres:
    conn: object

    def execute(self, sql: str, params: dict | None = None):
        with self.conn.cursor() as cur:
            cur.execute(sql, params or {})
            return cur.fetchall() if cur.description else None

    def upsert_json(self, table: str, record_id: str, payload: dict) -> None:
        self.execute(
            f"INSERT INTO {table}(id, payload) VALUES (%(id)s, %(payload)s) "
            f"ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload",
            {"id": record_id, "payload": payload},
        )
