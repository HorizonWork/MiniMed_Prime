"""Cypher-only KG conflict marker.

Companion to :mod:`kg_build.conflict_detector` (which expects an
in-memory ``AssertionRepo``). This module bypasses the repo entirely and
runs two Cypher writes against Neo4j to populate ``Assertion.conflict_score``:

  1. **Polarity flip**: same (subject, predicate, object) with opposite
     polarities → both assertions get conflict_score = 0.5.
  2. **Named-pair conflict**: assertions on the same (subject, object)
     where the predicate pair appears in
     :data:`conflict_detector.CONFLICTING_PREDICATES` (e.g.,
     ``treats`` vs ``contraindicated_for``) → both get
     conflict_score += 0.25 (capped at 1.0).

Required by ``reasoning.negative_sampler.NegativeSampler.generate_unsupported_paths``,
which queries ``a.conflict_score >= 0.5`` to mine negative training paths.
Without this CLI step, the PrimeKG-only KG has zero conflict_scores and
``unsupported_paths`` returns ``[]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from minimed_rag.kg_build.conflict_detector import CONFLICTING_PREDICATES

logger = logging.getLogger(__name__)


# Polarity-flip detection. The elementId(a1) < elementId(a2) clause
# halves the work and prevents double-counting (the symmetric query
# yields each ordered pair twice).
_POLARITY_FLIP_CYPHER = """
MATCH (s)<-[:SUBJECT]-(a1:Assertion)-[:OBJECT]->(o)
MATCH (s)<-[:SUBJECT]-(a2:Assertion)-[:OBJECT]->(o)
WHERE a1.is_current = true
  AND a2.is_current = true
  AND a1.predicate = a2.predicate
  AND a1.polarity <> a2.polarity
  AND elementId(a1) < elementId(a2)
SET a1.conflict_score = CASE
      WHEN coalesce(a1.conflict_score, 0.0) < 0.5 THEN 0.5
      ELSE a1.conflict_score
    END,
    a2.conflict_score = CASE
      WHEN coalesce(a2.conflict_score, 0.0) < 0.5 THEN 0.5
      ELSE a2.conflict_score
    END
RETURN count(a1) AS pairs
""".strip()


# Named-pair conflict detection. Iterates the CONFLICTING_PREDICATES
# tuples; each match adds 0.25 to both sides, capped at 1.0.
_NAMED_PAIR_CYPHER = """
UNWIND $pairs AS pair
MATCH (s)<-[:SUBJECT]-(a1:Assertion)-[:OBJECT]->(o)
MATCH (s)<-[:SUBJECT]-(a2:Assertion)-[:OBJECT]->(o)
WHERE a1.is_current = true AND a2.is_current = true
  AND a1.predicate = pair.left
  AND a2.predicate = pair.right
SET a1.conflict_score = CASE
      WHEN coalesce(a1.conflict_score, 0.0) + 0.25 > 1.0 THEN 1.0
      ELSE coalesce(a1.conflict_score, 0.0) + 0.25
    END,
    a2.conflict_score = CASE
      WHEN coalesce(a2.conflict_score, 0.0) + 0.25 > 1.0 THEN 1.0
      ELSE coalesce(a2.conflict_score, 0.0) + 0.25
    END
RETURN count(a1) AS pairs
""".strip()


# Reset clause for a clean re-run. Only touches assertions in the given
# graph_version when one is supplied.
_RESET_CYPHER = (
    "MATCH (a:Assertion) "
    "{filter}"
    "SET a.conflict_score = 0.0 "
    "RETURN count(a) AS reset"
)


@dataclass
class CypherConflictMarker:
    """Stamp ``Assertion.conflict_score`` for self-conflicting facts."""

    neo4j: Any
    counts: dict[str, int] = field(
        default_factory=lambda: {"reset": 0, "polarity_pairs": 0, "named_pairs": 0}
    )

    def reset(self, graph_version: str | None = None) -> int:
        """Zero out conflict_score on existing assertions before re-marking."""
        filter_clause = "WHERE a.graph_version = $gv " if graph_version else ""
        cypher = _RESET_CYPHER.format(filter=filter_clause)
        params = {"gv": graph_version} if graph_version else {}
        rows = self.neo4j.query(cypher, params)
        reset = self._scalar(rows, "reset")
        self.counts["reset"] = reset
        return reset

    def mark_polarity_flips(self) -> int:
        rows = self.neo4j.query(_POLARITY_FLIP_CYPHER, {})
        pairs = self._scalar(rows, "pairs")
        self.counts["polarity_pairs"] = pairs
        return pairs

    def mark_named_pairs(self) -> int:
        pairs_payload = [{"left": left, "right": right} for left, right in CONFLICTING_PREDICATES]
        rows = self.neo4j.query(_NAMED_PAIR_CYPHER, {"pairs": pairs_payload})
        pairs = self._scalar(rows, "pairs")
        self.counts["named_pairs"] = pairs
        return pairs

    def run(self, *, graph_version: str | None = None, reset_first: bool = True) -> dict[str, int]:
        if reset_first:
            self.reset(graph_version)
        self.mark_polarity_flips()
        self.mark_named_pairs()
        logger.info("cypher_conflict_marker: %s", self.counts)
        return dict(self.counts)

    @staticmethod
    def _scalar(rows: Any, key: str) -> int:
        if not rows:
            return 0
        first = rows[0]
        if isinstance(first, dict):
            return int(first.get(key) or 0)
        try:
            return int(first[0])
        except (TypeError, KeyError, IndexError):
            return 0


__all__ = ["CypherConflictMarker"]
