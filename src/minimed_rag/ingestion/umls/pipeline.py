"""UMLS streaming ingestion pipeline.

Three sequential passes over the RRF files placed at
``{settings.umls_data_dir}/{release}/``:

1. MRCONSO → :Concept + :Term nodes (Neo4j), `concept`/`term`/`umls_crosswalk`
   rows (Postgres). First-STR wins for concept.preferred_name; later preferred
   atoms refine via ``coalesce`` in the Cypher MERGE.
2. MRSTY → semantic_types / tuis arrays on existing :Concept nodes.
3. MRREL → reified :Assertion nodes linking two Concept nodes.

Postgres writes are OPTIONAL — pass ``postgres=None`` to skip them (useful
for unit tests and when the Postgres schema hasn't been bootstrapped).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from minimed_rag.common.ids import make_concept_id
from minimed_rag.ingestion.base import SourceIngestionPipeline
from minimed_rag.ingestion.umls.mrconso_parser import iter_mrconso
from minimed_rag.ingestion.umls.mrrel_parser import iter_mrrel
from minimed_rag.ingestion.umls.mrsty_parser import iter_mrsty
from minimed_rag.ingestion.umls.normalizer import (
    _make_term_id,
    _normalize_text,
    mrrel_row_to_records,
)
from minimed_rag.ingestion.umls.predicate_map import UMLSPredicateMapper

logger = logging.getLogger(__name__)


class UMLSIngestionPipeline(SourceIngestionPipeline):
    source_name = "umls"

    def __init__(
        self,
        neo4j_loader: Any,
        postgres: Any = None,
        predicate_mapper: UMLSPredicateMapper | None = None,
        data_dir: str | Path = "data/umls",
        graph_version: str = "kg_local",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.neo4j_loader = neo4j_loader
        self.postgres = postgres
        self.predicate_mapper = predicate_mapper or UMLSPredicateMapper()
        self.data_dir = Path(data_dir)
        self.graph_version = graph_version

    # ── Top-level orchestration ──────────────────────────────────────────────
    def run(
        self,
        release: str = "local",
        limit_mrconso: int | None = None,
        limit_mrsty: int | None = None,
        limit_mrrel: int | None = None,
        skip: Iterable[str] = (),
    ) -> dict[str, int]:
        skip_set = set(skip)
        if "mrconso" not in skip_set:
            self._ingest_mrconso(release, limit_mrconso)
        if "mrsty" not in skip_set:
            self._ingest_mrsty(release, limit_mrsty)
        if "mrrel" not in skip_set:
            self._ingest_mrrel(release, limit_mrrel)
        return self.neo4j_loader.counts_snapshot()

    def _release_dir(self, release: str) -> Path:
        return self.data_dir / release

    def _rrf_path(self, release: str, filename: str) -> Path:
        path = self._release_dir(release) / filename
        if not path.exists():
            raise FileNotFoundError(
                f"UMLS file not found: {path}. Place the RRF file there or override --data-dir."
            )
        return path

    # ── MRCONSO ──────────────────────────────────────────────────────────────
    def _ingest_mrconso(self, release: str, limit: int | None) -> None:
        path = self._rrf_path(release, "MRCONSO.RRF")
        logger.info("umls: streaming MRCONSO from %s (limit=%s)", path, limit)

        seen_cuis: set[str] = set()
        concept_neo_batch: list[dict] = []
        term_neo_batch: list[dict] = []
        concept_pg_batch: list[tuple] = []
        term_pg_batch: list[tuple] = []
        crosswalk_pg_batch: list[tuple] = []
        batch_size = self.neo4j_loader.batch_size
        rows = 0

        for row in iter_mrconso(path, limit=limit):
            cui = row["cui"]
            term_id = _make_term_id(row["aui"])
            concept_id = make_concept_id(cui)
            normalized = _normalize_text(row["str"])

            if cui not in seen_cuis:
                seen_cuis.add(cui)
                concept_neo_batch.append(
                    {
                        "cui": cui,
                        "preferred_name": row["str"] if row["is_preferred"] else row["str"],
                        "graph_version": self.graph_version,
                    }
                )
                concept_pg_batch.append((concept_id, cui, row["str"], self.graph_version))
            elif row["is_preferred"]:
                # Refine preferred_name on later preferred atoms.
                concept_neo_batch.append(
                    {
                        "cui": cui,
                        "preferred_name": row["str"],
                        "graph_version": self.graph_version,
                    }
                )

            term_neo_batch.append(
                {
                    "term_id": term_id,
                    "cui": cui,
                    "text": row["str"],
                    "normalized_text": normalized,
                    "language": row["lat"],
                    "source_system": row["sab"],
                    "is_preferred": bool(row["is_preferred"]),
                }
            )
            term_pg_batch.append(
                (term_id, concept_id, row["str"], normalized, row["lat"], row["sab"])
            )
            crosswalk_pg_batch.append((row["sab"], row["code"], cui))
            rows += 1

            if len(term_neo_batch) >= batch_size:
                self._flush_mrconso(
                    concept_neo_batch,
                    term_neo_batch,
                    concept_pg_batch,
                    term_pg_batch,
                    crosswalk_pg_batch,
                )
                concept_neo_batch = []
                term_neo_batch = []
                concept_pg_batch = []
                term_pg_batch = []
                crosswalk_pg_batch = []
            if rows % 500_000 == 0:
                logger.info("umls MRCONSO: %d rows, %d unique CUIs", rows, len(seen_cuis))

        self._flush_mrconso(
            concept_neo_batch,
            term_neo_batch,
            concept_pg_batch,
            term_pg_batch,
            crosswalk_pg_batch,
        )
        logger.info("umls MRCONSO: done (%d rows, %d unique CUIs)", rows, len(seen_cuis))

    def _flush_mrconso(
        self,
        concept_neo: list[dict],
        term_neo: list[dict],
        concept_pg: list[tuple],
        term_pg: list[tuple],
        crosswalk_pg: list[tuple],
    ) -> None:
        if concept_neo:
            self.neo4j_loader.upsert_concepts(iter(concept_neo))
        if term_neo:
            self.neo4j_loader.upsert_terms(iter(term_neo))
        if self.postgres is not None:
            if concept_pg:
                self._insert_concepts_pg(concept_pg)
            if term_pg:
                self._insert_terms_pg(term_pg)
            if crosswalk_pg:
                self._insert_crosswalk_pg(crosswalk_pg)
            self.postgres.commit()

    # ── MRSTY ────────────────────────────────────────────────────────────────
    def _ingest_mrsty(self, release: str, limit: int | None) -> None:
        path = self._rrf_path(release, "MRSTY.RRF")
        logger.info("umls: streaming MRSTY from %s (limit=%s)", path, limit)

        batch: list[dict] = []
        pg_batch: list[tuple] = []
        batch_size = self.neo4j_loader.batch_size
        rows = 0

        for row in iter_mrsty(path, limit=limit):
            batch.append({"cui": row["cui"], "tui": row["tui"], "sty": row["sty"]})
            pg_batch.append((row["cui"], row["tui"], row["sty"]))
            rows += 1
            if len(batch) >= batch_size:
                self.neo4j_loader.upsert_semantic_types(iter(batch))
                if self.postgres is not None:
                    self._insert_semantic_types_pg(pg_batch)
                    self.postgres.commit()
                batch = []
                pg_batch = []

        if batch:
            self.neo4j_loader.upsert_semantic_types(iter(batch))
        if self.postgres is not None and pg_batch:
            self._insert_semantic_types_pg(pg_batch)
            self.postgres.commit()
        logger.info("umls MRSTY: done (%d rows)", rows)

    # ── MRREL ────────────────────────────────────────────────────────────────
    def _ingest_mrrel(self, release: str, limit: int | None) -> None:
        path = self._rrf_path(release, "MRREL.RRF")
        logger.info("umls: streaming MRREL from %s (limit=%s)", path, limit)

        concept_stubs_seen: set[str] = set()
        concept_batch: list[dict] = []
        assertion_batch: list[dict] = []
        batch_size = self.neo4j_loader.batch_size
        rows = 0

        for row in iter_mrrel(path, limit=limit):
            for cui in (row["cui1"], row["cui2"]):
                if cui not in concept_stubs_seen:
                    concept_stubs_seen.add(cui)
                    concept_batch.append(
                        {
                            "cui": cui,
                            "preferred_name": None,
                            "graph_version": self.graph_version,
                        }
                    )
            for record in mrrel_row_to_records(
                row,
                self.predicate_mapper,
                release=release,
                graph_version=self.graph_version,
            ):
                assertion_batch.append(record.payload)
            rows += 1

            if len(assertion_batch) >= batch_size:
                if concept_batch:
                    self.neo4j_loader.upsert_concepts(iter(concept_batch))
                    concept_batch = []
                self.neo4j_loader.upsert_umls_assertions(iter(assertion_batch))
                assertion_batch = []
            if rows % 500_000 == 0:
                logger.info(
                    "umls MRREL: %d rows, %d unique CUIs stubbed",
                    rows,
                    len(concept_stubs_seen),
                )

        if concept_batch:
            self.neo4j_loader.upsert_concepts(iter(concept_batch))
        if assertion_batch:
            self.neo4j_loader.upsert_umls_assertions(iter(assertion_batch))
        logger.info("umls MRREL: done (%d rows)", rows)

    # ── Postgres bulk inserts (COPY → temp → INSERT ON CONFLICT) ─────────────
    def _insert_concepts_pg(self, rows: list[tuple]) -> None:
        with self.postgres.conn.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _concept_stage "
                "(concept_id TEXT, cui TEXT, preferred_name TEXT, graph_version TEXT) "
                "ON COMMIT DROP"
            )
            cur.execute("TRUNCATE _concept_stage")
            with cur.copy(
                "COPY _concept_stage (concept_id, cui, preferred_name, graph_version) FROM STDIN"
            ) as copy:
                for row in rows:
                    copy.write_row(row)
            cur.execute(
                "INSERT INTO concept (concept_id, cui, preferred_name, graph_version) "
                "SELECT DISTINCT ON (cui) concept_id, cui, preferred_name, graph_version "
                "FROM _concept_stage "
                "ON CONFLICT (cui) DO UPDATE "
                "SET preferred_name = EXCLUDED.preferred_name "
                "WHERE concept.preferred_name IS NULL OR concept.preferred_name = ''"
            )

    def _insert_terms_pg(self, rows: list[tuple]) -> None:
        with self.postgres.conn.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _term_stage "
                "(term_id TEXT, concept_id TEXT, text TEXT, normalized_text TEXT, "
                " language TEXT, source_system TEXT) ON COMMIT DROP"
            )
            cur.execute("TRUNCATE _term_stage")
            with cur.copy(
                "COPY _term_stage (term_id, concept_id, text, normalized_text, "
                "language, source_system) FROM STDIN"
            ) as copy:
                for row in rows:
                    copy.write_row(row)
            cur.execute(
                "INSERT INTO term (term_id, concept_id, text, normalized_text, "
                "language, source_system) "
                "SELECT term_id, concept_id, text, normalized_text, language, source_system "
                "FROM _term_stage "
                "ON CONFLICT (term_id) DO NOTHING"
            )

    def _insert_crosswalk_pg(self, rows: list[tuple]) -> None:
        with self.postgres.conn.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _crosswalk_stage "
                "(sab TEXT, code TEXT, cui TEXT) ON COMMIT DROP"
            )
            cur.execute("TRUNCATE _crosswalk_stage")
            with cur.copy("COPY _crosswalk_stage (sab, code, cui) FROM STDIN") as copy:
                for row in rows:
                    copy.write_row(row)
            cur.execute(
                "INSERT INTO umls_crosswalk (sab, code, cui) "
                "SELECT DISTINCT sab, code, cui FROM _crosswalk_stage "
                "ON CONFLICT DO NOTHING"
            )

    def _insert_semantic_types_pg(self, rows: list[tuple]) -> None:
        with self.postgres.conn.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _sty_stage "
                "(cui TEXT, tui TEXT, sty TEXT) ON COMMIT DROP"
            )
            cur.execute("TRUNCATE _sty_stage")
            with cur.copy("COPY _sty_stage (cui, tui, sty) FROM STDIN") as copy:
                for row in rows:
                    copy.write_row(row)
            cur.execute(
                "INSERT INTO concept_semantic_type (cui, tui, sty) "
                "SELECT DISTINCT cui, tui, sty FROM _sty_stage "
                "ON CONFLICT DO NOTHING"
            )

    # ── Streaming helpers (for tests / future reuse) ─────────────────────────
    def stream_mrconso(self, release: str = "local", limit: int | None = None) -> Iterator[dict]:
        return iter_mrconso(self._rrf_path(release, "MRCONSO.RRF"), limit=limit)

    def stream_mrsty(self, release: str = "local", limit: int | None = None) -> Iterator[dict]:
        return iter_mrsty(self._rrf_path(release, "MRSTY.RRF"), limit=limit)

    def stream_mrrel(self, release: str = "local", limit: int | None = None) -> Iterator[dict]:
        return iter_mrrel(self._rrf_path(release, "MRREL.RRF"), limit=limit)

    # ── Legacy SourceIngestionPipeline adapters (unused by run()) ────────────
    def download(self):
        return []

    def parse(self, raw_file):
        return []

    def normalize(self, record):
        return []
