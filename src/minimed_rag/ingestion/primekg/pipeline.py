"""PrimeKG ingestion pipeline — streaming CSV → Neo4j reified schema."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from minimed_rag.ingestion.base import NormalizedRecord, RawRecord, SourceIngestionPipeline
from minimed_rag.ingestion.primekg.normalizer import normalize_primekg_edge
from minimed_rag.ingestion.primekg.parser import (
    RawPrimeKGEdge,
    iter_primekg_edges,
    parse_primekg_edges,
)
from minimed_rag.ingestion.primekg.predicate_map import SimplePredicateMapper

logger = logging.getLogger(__name__)


class PrimeKGIngestionPipeline(SourceIngestionPipeline):
    """Stream PrimeKG edges from disk directly into Neo4j.

    Differences from the generic ``SourceIngestionPipeline.run()``:

    - Reads CSV row-by-row rather than materialising ``RawFile.bytes``.
    - Deduplicates entities in-memory (~130k unique PrimeKG nodes → ~26 MB).
    - Upserts entities before their dependent assertions within each batch
      so the assertion-MATCH always finds its endpoints.
    """

    source_name = "primekg"

    def __init__(
        self,
        downloader,
        predicate_mapper: SimplePredicateMapper | None = None,
        neo4j_loader: Any = None,
        graph_version: str = "kg_local",
        predicate_registry: Any = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.downloader = downloader
        self.predicate_mapper = predicate_mapper or SimplePredicateMapper()
        self.neo4j_loader = neo4j_loader
        self.graph_version = graph_version
        self.predicate_registry = predicate_registry

    # ── Streaming ingest ─────────────────────────────────────────────────────
    def run(
        self,
        release: str = "local",
        limit: int | None = None,
        csv_path: str | Path | None = None,
    ) -> dict[str, int]:
        """Full pipeline: download (if needed) → stream → upsert → report counts."""
        if self.neo4j_loader is None:
            raise RuntimeError(
                "PrimeKGIngestionPipeline.run() requires a neo4j_loader. "
                "Instantiate the pipeline with a Neo4jLoader or call the "
                "lower-level stream_records() helper directly."
            )

        path = Path(csv_path) if csv_path else self.downloader.ensure_download(release)
        logger.info("primekg: streaming ingest from %s (limit=%s)", path, limit)

        entity_ids_seen: set[str] = set()
        entity_batch: list[dict] = []
        assertion_batch: list[dict] = []
        batch_size = self.neo4j_loader.batch_size
        rows_processed = 0

        for edge in iter_primekg_edges(path, limit=limit):
            for record in normalize_primekg_edge(
                edge,
                self.predicate_mapper,
                source_release=release,
                graph_version=self.graph_version,
                predicate_registry=self.predicate_registry,
            ):
                if record.record_type == "source_entity":
                    eid = record.payload["entity_id"]
                    if eid not in entity_ids_seen:
                        entity_batch.append(record.payload)
                        entity_ids_seen.add(eid)
                elif record.record_type == "source_assertion":
                    assertion_batch.append(record.payload)

            rows_processed += 1
            if len(assertion_batch) >= batch_size:
                self._flush(entity_batch, assertion_batch)
                entity_batch = []
                assertion_batch = []
            if rows_processed % 100_000 == 0:
                logger.info(
                    "primekg: %d rows processed, %d unique entities",
                    rows_processed,
                    len(entity_ids_seen),
                )

        if entity_batch or assertion_batch:
            self._flush(entity_batch, assertion_batch)

        counts = self.neo4j_loader.counts_snapshot()
        logger.info(
            "primekg: done. %d rows, %d unique entities. counts=%s",
            rows_processed,
            len(entity_ids_seen),
            counts,
        )
        return counts

    def _flush(self, entities: list[dict], assertions: list[dict]) -> None:
        if entities:
            self.neo4j_loader.upsert_entities(iter(entities))
        if assertions:
            self.neo4j_loader.upsert_primekg_assertions(iter(assertions))

    # ── Legacy SourceIngestionPipeline hooks (small-input paths only) ────────
    def download(self):
        return self.downloader.download_release("local") if self.downloader else []

    def parse(self, raw_file):
        return [RawRecord("PrimeKGEdge", edge.__dict__) for edge in parse_primekg_edges(raw_file)]

    def normalize(self, record: RawRecord) -> list[NormalizedRecord]:
        return normalize_primekg_edge(
            RawPrimeKGEdge(**record.payload),
            self.predicate_mapper,
            source_release="local",
            graph_version=self.graph_version,
            predicate_registry=self.predicate_registry,
        )

    def stream_records(
        self,
        release: str = "local",
        limit: int | None = None,
    ) -> Iterator[NormalizedRecord]:
        """Lower-level streaming helper for callers that want raw records
        without the Neo4j side effects (e.g., tests, Postgres mirroring)."""
        path = self.downloader.ensure_download(release)
        for edge in iter_primekg_edges(path, limit=limit):
            yield from normalize_primekg_edge(
                edge,
                self.predicate_mapper,
                source_release=release,
                graph_version=self.graph_version,
                predicate_registry=self.predicate_registry,
            )
