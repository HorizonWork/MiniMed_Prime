"""End-to-end unit test for the PrimeKG streaming pipeline.

Uses a recording Neo4j mock so the test runs with no external services.
Exercises the single-pass dedup/batching loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from minimed_rag.ingestion.primekg.downloader import PrimeKGDownloader
from minimed_rag.ingestion.primekg.pipeline import PrimeKGIngestionPipeline
from minimed_rag.ingestion.primekg.predicate_map import SimplePredicateMapper
from minimed_rag.kg_build.neo4j_loader import Neo4jLoader

CSV = (
    "x_id,x_name,x_type,relation,y_id,y_name,y_type,source\n"
    "DB:1,Metformin,drug,indication,MONDO:1,Diabetes,disease,curated\n"
    "DB:1,Metformin,drug,drug_protein,NCBI:1,AMPK,gene/protein,curated\n"
    "DB:2,Aspirin,drug,contraindication,MONDO:2,GastricUlcer,disease,curated\n"
    "MONDO:1,Diabetes,disease,disease_protein,NCBI:2,INS,gene/protein,curated\n"
    "MONDO:1,Diabetes,disease,disease_phenotype_positive,HP:1,Polyuria,phenotype,curated\n"
)


@dataclass
class RecordingNeo4j:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def run(self, cypher: str, **params: Any) -> list:
        self.calls.append((cypher, params))
        return []


class InMemoryDownloader(PrimeKGDownloader):
    """Skips network; returns a pre-staged CSV path."""

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path
        self.data_dir = csv_path.parent
        self.url = "file://" + str(csv_path)
        self.force = False

    def ensure_download(self, release: str) -> Path:
        return self._csv_path


@pytest.fixture
def csv_path(tmp_path: Path) -> Path:
    path = tmp_path / "kg.csv"
    path.write_text(CSV, encoding="utf-8")
    return path


def test_pipeline_streams_entities_and_assertions(csv_path: Path):
    neo4j = RecordingNeo4j()
    loader = Neo4jLoader(neo4j=neo4j, graph_version="kg_test", batch_size=2)
    pipeline = PrimeKGIngestionPipeline(
        downloader=InMemoryDownloader(csv_path),
        predicate_mapper=SimplePredicateMapper(),
        neo4j_loader=loader,
        graph_version="kg_test",
    )

    counts = pipeline.run(release="local")

    # 5 edges in the fixture. 2 PrimeKG edges per row produce 2 entities
    # each (subject+object), but entity IDs dedupe:
    #   DB:1 Metformin (Drug), MONDO:1 Diabetes (Disease),
    #   NCBI:1 AMPK (GeneProtein), DB:2 Aspirin (Drug),
    #   MONDO:2 GastricUlcer (Disease), NCBI:2 INS (GeneProtein),
    #   HP:1 Polyuria (Phenotype) = 7 unique entities.
    assert counts["entities"] == 7
    assert counts["assertions_entity"] == 5


def test_pipeline_limit_flag_stops_early(csv_path: Path):
    neo4j = RecordingNeo4j()
    loader = Neo4jLoader(neo4j=neo4j, graph_version="kg_test", batch_size=5)
    pipeline = PrimeKGIngestionPipeline(
        downloader=InMemoryDownloader(csv_path),
        predicate_mapper=SimplePredicateMapper(),
        neo4j_loader=loader,
    )
    counts = pipeline.run(release="local", limit=2)
    assert counts["assertions_entity"] == 2


def test_pipeline_requires_loader():
    pipeline = PrimeKGIngestionPipeline(
        downloader=None,
        predicate_mapper=SimplePredicateMapper(),
        neo4j_loader=None,
    )
    with pytest.raises(RuntimeError, match="requires a neo4j_loader"):
        pipeline.run()


def test_stream_records_yields_without_loader(csv_path: Path):
    pipeline = PrimeKGIngestionPipeline(
        downloader=InMemoryDownloader(csv_path),
        predicate_mapper=SimplePredicateMapper(),
        neo4j_loader=None,  # not needed for streaming
    )
    records = list(pipeline.stream_records(release="local"))
    # 5 rows × (2 entity + 1 assertion) = 15 records
    assert len(records) == 15
    types = [r.record_type for r in records]
    assert types.count("source_entity") == 10
    assert types.count("source_assertion") == 5
