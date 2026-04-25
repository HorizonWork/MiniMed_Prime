"""Unit tests for the UMLS↔PrimeKG crosswalk runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from minimed_rag.kg_build.umls_crosswalk import CrosswalkRunner


@dataclass
class FakeNeo4j:
    entities: list[dict]
    write_calls: list[tuple[str, dict]] = field(default_factory=list)

    def query(self, cypher: str, params: dict) -> list[dict]:
        return list(self.entities)

    def run(self, cypher: str, **params: Any) -> list:
        self.write_calls.append((cypher, params))
        return []


@dataclass
class FakePostgres:
    crosswalk: dict[tuple[str, str], str]
    queries: list[tuple[str, dict]] = field(default_factory=list)

    def execute(self, sql: str, params: dict) -> list[tuple]:
        self.queries.append((sql, params))
        cui = self.crosswalk.get((params["sab"], params["code"]))
        return [(cui,)] if cui else []


def test_crosswalk_matches_mondo_and_drugbank():
    entities = [
        {
            "entity_id": "KG:Disease:h1",
            "entity_type": "Disease",
            "source_id": "MONDO:0005148",
        },
        {
            "entity_id": "KG:Drug:h2",
            "entity_type": "Drug",
            "source_id": "DB00331",
        },
        {
            "entity_id": "KG:GeneProtein:h3",
            "entity_type": "GeneProtein",
            "source_id": "5292",  # NCBI gene (digit string)
        },
        {
            "entity_id": "KG:Drug:h4",
            "entity_type": "Drug",
            "source_id": "WEIRD_ID",  # no mapping
        },
    ]
    crosswalk = {
        ("MONDO", "MONDO:0005148"): "C0011849",
        ("DRUGBANK", "DB00331"): "C0025598",
        ("NCBI", "5292"): "C0000001",
    }
    neo4j = FakeNeo4j(entities=entities)
    postgres = FakePostgres(crosswalk=crosswalk)

    runner = CrosswalkRunner(neo4j=neo4j, postgres=postgres, batch_size=10)
    counts = runner.run()

    assert counts["examined"] == 4
    assert counts["matched"] == 3
    assert counts["linked"] == 3
    assert len(neo4j.write_calls) == 1
    flush_rows = neo4j.write_calls[0][1]["rows"]
    linked_cuis = {r["cui"] for r in flush_rows}
    assert linked_cuis == {"C0011849", "C0025598", "C0000001"}


def test_crosswalk_respects_batch_size():
    entities = [
        {"entity_id": f"KG:Disease:{i}", "entity_type": "Disease", "source_id": f"MONDO:{i}"}
        for i in range(5)
    ]
    crosswalk = {("MONDO", f"MONDO:{i}"): f"C{i}" for i in range(5)}
    neo4j = FakeNeo4j(entities=entities)
    postgres = FakePostgres(crosswalk=crosswalk)
    runner = CrosswalkRunner(neo4j=neo4j, postgres=postgres, batch_size=2)
    runner.run()
    # 5 matched / batch 2 → 3 flushes (2, 2, 1)
    assert len(neo4j.write_calls) == 3
    assert [len(c[1]["rows"]) for c in neo4j.write_calls] == [2, 2, 1]


def test_crosswalk_no_matches_returns_counts():
    entities = [{"entity_id": "x", "entity_type": "Drug", "source_id": "unknown"}]
    neo4j = FakeNeo4j(entities=entities)
    postgres = FakePostgres(crosswalk={})
    runner = CrosswalkRunner(neo4j=neo4j, postgres=postgres)
    counts = runner.run()
    assert counts["examined"] == 1
    assert counts["matched"] == 0
    assert counts["linked"] == 0
    assert neo4j.write_calls == []
