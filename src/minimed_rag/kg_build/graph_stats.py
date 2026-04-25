"""Per-entity degree statistics derived from the reified-Assertion KG.

Computes ``degree`` (total assertions touching the entity), ``in_degree``
(entity acts as Assertion OBJECT), and ``out_degree`` (entity acts as
Assertion SUBJECT) for every :Entity node, and persists to Postgres
``entity_stats``. The PathScorer reads these stats via
:class:`EntityStatsRepo` to apply ``hub_penalty`` — paths through
high-degree entities are demoted because they're typically generic hubs
(e.g., "Disease", "Cancer") rather than question-specific intermediaries.

Snapshot-keyed: each run is tagged with a caller-supplied ``snapshot_id``
(usually matches ``graph_version``) so we can re-compute and switch
without losing the previous baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS entity_stats (
    entity_id   TEXT        NOT NULL,
    degree      INTEGER     NOT NULL,
    in_degree   INTEGER     NOT NULL,
    out_degree  INTEGER     NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    snapshot_id TEXT        NOT NULL,
    PRIMARY KEY (entity_id, snapshot_id)
)
"""


_DEGREE_CYPHER = """
MATCH (e:Entity)
OPTIONAL MATCH (a:Assertion)-[r:SUBJECT|OBJECT]->(e)
WHERE a.is_current = true
RETURN e.entity_id AS entity_id,
       count(r) AS degree,
       sum(CASE WHEN type(r) = 'OBJECT' THEN 1 ELSE 0 END) AS in_degree,
       sum(CASE WHEN type(r) = 'SUBJECT' THEN 1 ELSE 0 END) AS out_degree
"""


_INSERT_SQL = """
INSERT INTO entity_stats (entity_id, degree, in_degree, out_degree, snapshot_id)
VALUES (%(entity_id)s, %(degree)s, %(in_degree)s, %(out_degree)s, %(snapshot_id)s)
"""


_DELETE_SNAPSHOT_SQL = "DELETE FROM entity_stats WHERE snapshot_id = %(snapshot_id)s"


@dataclass
class GraphStatsBuilder:
    """Compute per-entity assertion-degree counts and persist a snapshot."""

    neo4j: Any
    postgres: Any
    batch_size: int = 1000
    counts: dict[str, int] = field(
        default_factory=lambda: {"examined": 0, "written": 0}
    )

    def _ensure_schema(self) -> None:
        self.postgres.execute(_CREATE_TABLE_SQL)
        self.postgres.commit()

    def compute_and_persist(self, snapshot_id: str) -> dict[str, int]:
        """Replace the snapshot's rows with freshly-computed degree counts."""
        if not snapshot_id:
            raise ValueError("snapshot_id is required")
        self._ensure_schema()
        self.postgres.execute(_DELETE_SNAPSHOT_SQL, {"snapshot_id": snapshot_id})

        batch: list[dict] = []
        for row in self.neo4j.query(_DEGREE_CYPHER, {}):
            entity_id = row.get("entity_id")
            if not entity_id:
                continue
            self.counts["examined"] += 1
            batch.append(
                {
                    "entity_id": entity_id,
                    "degree": int(row.get("degree") or 0),
                    "in_degree": int(row.get("in_degree") or 0),
                    "out_degree": int(row.get("out_degree") or 0),
                    "snapshot_id": snapshot_id,
                }
            )
            if len(batch) >= self.batch_size:
                self._flush(batch)
                batch = []
        if batch:
            self._flush(batch)
        self.postgres.commit()
        logger.info("graph_stats snapshot %s: %s", snapshot_id, self.counts)
        return dict(self.counts)

    def _flush(self, rows: list[dict]) -> None:
        self.postgres.execute_many(_INSERT_SQL, rows)
        self.counts["written"] += len(rows)
