"""Unit tests for the GraphStatsBuilder + EntityStatsRepo pair."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from minimed_rag.kg_build.graph_stats import GraphStatsBuilder
from minimed_rag.storage.repositories.entity_stats_repo import EntityStatsRepo


@dataclass
class FakeNeo4j:
    rows: list[dict]
    last_cypher: str = ""

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        self.last_cypher = cypher
        return list(self.rows)


@dataclass
class FakePostgres:
    """In-memory stand-in that tracks every call. Each ``execute`` returns a
    pre-canned result keyed by SQL fragment substring (kept tiny for tests)."""

    rows_by_query: dict[str, list[tuple]] = field(default_factory=dict)
    executes: list[tuple[str, Any]] = field(default_factory=list)
    execute_many_calls: list[tuple[str, list[dict]]] = field(default_factory=list)
    commits: int = 0

    def execute(self, sql: str, params: Any = None) -> list[tuple] | None:
        self.executes.append((sql, params))
        for fragment, rows in self.rows_by_query.items():
            if fragment in sql:
                return list(rows)
        return None

    def execute_many(self, sql: str, rows: list[dict]) -> None:
        self.execute_many_calls.append((sql, list(rows)))

    def commit(self) -> None:
        self.commits += 1


def test_compute_and_persist_writes_per_entity_rows() -> None:
    rows = [
        {"entity_id": "E1", "degree": 5, "in_degree": 2, "out_degree": 3},
        {"entity_id": "E2", "degree": 1, "in_degree": 1, "out_degree": 0},
    ]
    neo4j = FakeNeo4j(rows=rows)
    postgres = FakePostgres()

    builder = GraphStatsBuilder(neo4j=neo4j, postgres=postgres, batch_size=10)
    counts = builder.compute_and_persist(snapshot_id="kg_local_test")

    assert counts == {"examined": 2, "written": 2}
    # CREATE TABLE + DELETE snapshot rows
    create_sqls = [sql for sql, _ in postgres.executes if "CREATE TABLE" in sql]
    delete_sqls = [sql for sql, _ in postgres.executes if "DELETE FROM" in sql]
    assert len(create_sqls) == 1
    assert len(delete_sqls) == 1
    # Bulk insert in one batch
    assert len(postgres.execute_many_calls) == 1
    inserted = postgres.execute_many_calls[0][1]
    assert {r["entity_id"] for r in inserted} == {"E1", "E2"}
    assert all(r["snapshot_id"] == "kg_local_test" for r in inserted)
    # Commits: schema + final
    assert postgres.commits >= 1


def test_compute_and_persist_respects_batch_size() -> None:
    rows = [
        {"entity_id": f"E{i}", "degree": i, "in_degree": 0, "out_degree": i}
        for i in range(5)
    ]
    neo4j = FakeNeo4j(rows=rows)
    postgres = FakePostgres()
    builder = GraphStatsBuilder(neo4j=neo4j, postgres=postgres, batch_size=2)
    builder.compute_and_persist(snapshot_id="snap1")
    # 5 rows / batch 2 → 3 flushes (2, 2, 1)
    assert [len(call[1]) for call in postgres.execute_many_calls] == [2, 2, 1]


def test_compute_and_persist_rejects_empty_snapshot_id() -> None:
    builder = GraphStatsBuilder(neo4j=FakeNeo4j(rows=[]), postgres=FakePostgres())
    with pytest.raises(ValueError):
        builder.compute_and_persist(snapshot_id="")


def test_compute_and_persist_skips_rows_with_empty_entity_id() -> None:
    rows = [
        {"entity_id": "", "degree": 5, "in_degree": 2, "out_degree": 3},
        {"entity_id": "E1", "degree": 5, "in_degree": 2, "out_degree": 3},
    ]
    neo4j = FakeNeo4j(rows=rows)
    postgres = FakePostgres()
    builder = GraphStatsBuilder(neo4j=neo4j, postgres=postgres)
    counts = builder.compute_and_persist(snapshot_id="snap")
    assert counts == {"examined": 1, "written": 1}


def test_repo_get_degree_returns_int() -> None:
    postgres = FakePostgres(rows_by_query={"SELECT degree FROM entity_stats": [(42,)]})
    repo = EntityStatsRepo(postgres=postgres)
    assert repo.get_degree("E1", "snap") == 42


def test_repo_get_degree_missing_returns_none() -> None:
    postgres = FakePostgres(rows_by_query={"SELECT degree FROM entity_stats": []})
    repo = EntityStatsRepo(postgres=postgres)
    assert repo.get_degree("missing", "snap") is None


def test_repo_get_p50_returns_float() -> None:
    postgres = FakePostgres(
        rows_by_query={"percentile_cont": [(8.0,)]}
    )
    repo = EntityStatsRepo(postgres=postgres)
    assert repo.get_p50_degree("snap") == 8.0


def test_repo_get_p50_empty_snapshot_returns_zero() -> None:
    postgres = FakePostgres(rows_by_query={"percentile_cont": [(None,)]})
    repo = EntityStatsRepo(postgres=postgres)
    assert repo.get_p50_degree("snap") == 0.0


def test_repo_get_top_degree_entities_returns_ids() -> None:
    postgres = FakePostgres(
        rows_by_query={"percent_rank": [("E1",), ("E2",), ("E3",)]}
    )
    repo = EntityStatsRepo(postgres=postgres)
    assert repo.get_top_degree_entities("snap", top_pct=0.01) == ["E1", "E2", "E3"]
