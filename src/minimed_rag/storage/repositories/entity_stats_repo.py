"""Read-side repository for per-entity degree stats (Postgres ``entity_stats``).

Companion to :mod:`minimed_rag.kg_build.graph_stats` (write side). The
PathScorer's ``hub_penalty`` and the NegativeSampler's
``generate_generic_hub_paths`` query through this repo. All methods take
a ``snapshot_id`` so callers can pin to a known baseline rather than
silently drifting when stats are recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EntityStatsRepo:
    postgres: Any

    def get_degree(self, entity_id: str, snapshot_id: str) -> int | None:
        rows = self.postgres.execute(
            "SELECT degree FROM entity_stats "
            "WHERE entity_id = %(entity_id)s AND snapshot_id = %(snapshot_id)s "
            "LIMIT 1",
            {"entity_id": entity_id, "snapshot_id": snapshot_id},
        )
        if not rows:
            return None
        return int(rows[0][0])

    def get_p50_degree(self, snapshot_id: str) -> float:
        """Median ``degree`` across non-leaf entities (degree > 1) in the snapshot.

        Excludes leaves so the median tracks the typical "interior" entity,
        which is what the hub-penalty formula calibrates against. Returns
        0.0 if the snapshot is empty (the caller should treat that as
        "no hub penalty applies").
        """
        rows = self.postgres.execute(
            "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY degree) "
            "FROM entity_stats "
            "WHERE snapshot_id = %(snapshot_id)s AND degree > 1",
            {"snapshot_id": snapshot_id},
        )
        if not rows or rows[0][0] is None:
            return 0.0
        return float(rows[0][0])

    def get_top_degree_entities(
        self, snapshot_id: str, top_pct: float = 0.01, limit: int = 1000
    ) -> list[str]:
        """Top-``top_pct`` entities by degree within the snapshot.

        Used by ``generate_generic_hub_paths`` as the seed set of "hubs" we
        want to walk through to produce uninformative-but-plausible
        negatives. ``limit`` guards against pathological cases.
        """
        rows = self.postgres.execute(
            "WITH ranked AS ("
            "  SELECT entity_id, "
            "         percent_rank() OVER (ORDER BY degree DESC) AS pr "
            "  FROM entity_stats WHERE snapshot_id = %(snapshot_id)s"
            ") "
            "SELECT entity_id FROM ranked WHERE pr <= %(pct)s LIMIT %(limit)s",
            {"snapshot_id": snapshot_id, "pct": top_pct, "limit": limit},
        )
        return [row[0] for row in (rows or [])]
