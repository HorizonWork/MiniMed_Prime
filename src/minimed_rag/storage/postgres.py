"""PostgreSQL metadata adapter."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Postgres:
    conn: Any

    def execute(self, sql: str, params: dict | Any = None) -> list | None:
        with self.conn.cursor() as cur:
            cur.execute(sql, params or {})
            return cur.fetchall() if cur.description else None

    def execute_many(self, sql: str, rows: list[dict] | list[tuple]) -> None:
        if not rows:
            return
        with self.conn.cursor() as cur:
            cur.executemany(sql, rows)

    def upsert_json(self, table: str, record_id: str, payload: dict) -> None:
        self.execute(
            f"INSERT INTO {table}(id, payload) VALUES (%(id)s, %(payload)s) "
            f"ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload",
            {"id": record_id, "payload": payload},
        )

    def copy_from(
        self,
        table: str,
        columns: list[str],
        rows: Iterable[tuple],
    ) -> int:
        """Bulk COPY FROM STDIN via psycopg3. Returns rows written."""
        columns_csv = ", ".join(columns)
        count = 0
        with self.conn.cursor() as cur:
            with cur.copy(f"COPY {table} ({columns_csv}) FROM STDIN") as copy:
                for row in rows:
                    copy.write_row(row)
                    count += 1
        return count

    def commit(self) -> None:
        self.conn.commit()

    def rollback(self) -> None:
        self.conn.rollback()

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()


def connect(dsn: str) -> Postgres:
    """Construct a Postgres adapter with a live psycopg3 connection."""
    import psycopg

    conn = psycopg.connect(dsn, autocommit=False)
    return Postgres(conn=conn)
