"""Unit tests for the CypherConflictMarker.

Verifies the two Cypher writes (polarity-flip + named-pair) fire with the
right parameters and that the public ``run()`` orchestrator chains
reset → polarity → named-pair correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from minimed_rag.kg_build.conflict_detector import CONFLICTING_PREDICATES
from minimed_rag.kg_build.cypher_conflict_marker import CypherConflictMarker


@dataclass
class FakeNeo4j:
    """Records every query and returns a canned scalar count."""

    canned: dict[str, list[dict]] = field(default_factory=dict)
    calls: list[tuple[str, dict]] = field(default_factory=list)

    def query(self, cypher: str, params: dict | None = None) -> list[dict]:
        self.calls.append((cypher, dict(params or {})))
        for fragment, rows in self.canned.items():
            if fragment in cypher:
                return list(rows)
        return []


def test_reset_no_filter_when_graph_version_omitted() -> None:
    neo4j = FakeNeo4j(canned={"SET a.conflict_score = 0.0": [{"reset": 7}]})
    marker = CypherConflictMarker(neo4j=neo4j)
    n = marker.reset()
    assert n == 7
    assert marker.counts["reset"] == 7
    sql, params = neo4j.calls[0]
    assert "SET a.conflict_score = 0.0" in sql
    assert "graph_version" not in sql  # no filter
    assert params == {}


def test_reset_filters_by_graph_version() -> None:
    neo4j = FakeNeo4j(canned={"SET a.conflict_score = 0.0": [{"reset": 3}]})
    marker = CypherConflictMarker(neo4j=neo4j)
    marker.reset(graph_version="kg_local")
    sql, params = neo4j.calls[0]
    assert "WHERE a.graph_version = $gv" in sql
    assert params == {"gv": "kg_local"}


def test_mark_polarity_flips_uses_elementid_dedup() -> None:
    """The polarity-flip Cypher must use elementId(a1) < elementId(a2) to
    avoid double-counting symmetric pairs."""
    neo4j = FakeNeo4j(canned={"a1.polarity <> a2.polarity": [{"pairs": 12}]})
    marker = CypherConflictMarker(neo4j=neo4j)
    n = marker.mark_polarity_flips()
    assert n == 12
    assert marker.counts["polarity_pairs"] == 12
    sql, _ = neo4j.calls[0]
    assert "elementId(a1) < elementId(a2)" in sql
    assert "a1.polarity <> a2.polarity" in sql


def test_mark_polarity_flips_caps_at_existing_score() -> None:
    """Idempotency: re-running should not raise existing conflict_score
    above 0.5 unless a higher value was already set."""
    neo4j = FakeNeo4j(canned={"a1.polarity <> a2.polarity": [{"pairs": 0}]})
    marker = CypherConflictMarker(neo4j=neo4j)
    marker.mark_polarity_flips()
    sql, _ = neo4j.calls[0]
    assert "WHEN coalesce(a1.conflict_score, 0.0) < 0.5 THEN 0.5" in sql


def test_mark_named_pairs_passes_conflict_pairs() -> None:
    neo4j = FakeNeo4j(canned={"UNWIND $pairs": [{"pairs": 4}]})
    marker = CypherConflictMarker(neo4j=neo4j)
    n = marker.mark_named_pairs()
    assert n == 4
    assert marker.counts["named_pairs"] == 4
    _, params = neo4j.calls[0]
    sent = params["pairs"]
    expected = [{"left": l, "right": r} for l, r in CONFLICTING_PREDICATES]
    assert sent == expected


def test_mark_named_pairs_cumulative_with_cap() -> None:
    neo4j = FakeNeo4j(canned={"UNWIND $pairs": [{"pairs": 0}]})
    marker = CypherConflictMarker(neo4j=neo4j)
    marker.mark_named_pairs()
    sql, _ = neo4j.calls[0]
    assert "coalesce(a1.conflict_score, 0.0) + 0.25" in sql
    assert "WHEN coalesce(a1.conflict_score, 0.0) + 0.25 > 1.0 THEN 1.0" in sql


def test_run_chains_reset_polarity_named_in_order() -> None:
    neo4j = FakeNeo4j(
        canned={
            "SET a.conflict_score = 0.0": [{"reset": 100}],
            "a1.polarity <> a2.polarity": [{"pairs": 5}],
            "UNWIND $pairs": [{"pairs": 2}],
        }
    )
    marker = CypherConflictMarker(neo4j=neo4j)
    counts = marker.run(graph_version="kg_local")
    assert counts == {"reset": 100, "polarity_pairs": 5, "named_pairs": 2}
    # Order: reset → polarity → named
    cyphers = [call[0] for call in neo4j.calls]
    assert "SET a.conflict_score = 0.0" in cyphers[0]
    assert "a1.polarity <> a2.polarity" in cyphers[1]
    assert "UNWIND $pairs" in cyphers[2]


def test_run_skips_reset_when_no_reset_true() -> None:
    neo4j = FakeNeo4j(
        canned={
            "a1.polarity <> a2.polarity": [{"pairs": 1}],
            "UNWIND $pairs": [{"pairs": 0}],
        }
    )
    marker = CypherConflictMarker(neo4j=neo4j)
    marker.run(reset_first=False)
    cyphers = [call[0] for call in neo4j.calls]
    assert not any("SET a.conflict_score = 0.0" in c for c in cyphers)
    assert marker.counts["reset"] == 0


def test_scalar_helper_handles_missing_rows() -> None:
    neo4j = FakeNeo4j(canned={})  # all queries return []
    marker = CypherConflictMarker(neo4j=neo4j)
    assert marker.reset() == 0
    assert marker.mark_polarity_flips() == 0
    assert marker.mark_named_pairs() == 0


def test_scalar_helper_handles_tuple_rows() -> None:
    neo4j = FakeNeo4j(canned={"SET a.conflict_score = 0.0": [(42,)]})
    marker = CypherConflictMarker(neo4j=neo4j)
    assert marker.reset() == 42
