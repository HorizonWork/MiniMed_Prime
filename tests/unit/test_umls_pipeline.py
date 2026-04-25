"""End-to-end test for the UMLS streaming pipeline.

Uses a recording Neo4j and a stub Postgres (set to None → skip Postgres).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from minimed_rag.ingestion.umls.pipeline import UMLSIngestionPipeline
from minimed_rag.ingestion.umls.predicate_map import UMLSPredicateMapper
from minimed_rag.kg_build.neo4j_loader import Neo4jLoader

MRCONSO_LINES = (
    "C0025598|ENG|P|L1|PF|S1|Y|A0001|||DB00331|DRUGBANK|IN|DB00331|Metformin|0|N|\n"
    "C0025598|ENG|S|L2|PF|S2|N|A0002|||DB00331|DRUGBANK|SY|DB00331|Metformin hydrochloride|0|N|\n"
    "C0011849|ENG|P|L3|PF|S3|Y|A0003|||MONDO:0005148|MONDO|PN|MONDO:0005148|Type 2 diabetes mellitus|0|N|\n"
)

MRSTY_LINES = (
    "C0025598|T121|A|Pharmacologic Substance|AT1|\nC0011849|T047|A|Disease or Syndrome|AT2|\n"
)

MRREL_LINES = "C0025598|A0001|CUI|RO|C0011849|A0003|CUI|may_treat|R001||DRUGBANK|DRUGBANK|||N|\n"


@dataclass
class RecordingNeo4j:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def run(self, cypher: str, **params: Any) -> list:
        self.calls.append((cypher, params))
        return []


@pytest.fixture
def umls_dir(tmp_path: Path) -> Path:
    release_dir = tmp_path / "local"
    release_dir.mkdir()
    (release_dir / "MRCONSO.RRF").write_text(MRCONSO_LINES, encoding="utf-8")
    (release_dir / "MRSTY.RRF").write_text(MRSTY_LINES, encoding="utf-8")
    (release_dir / "MRREL.RRF").write_text(MRREL_LINES, encoding="utf-8")
    return tmp_path


def test_pipeline_loads_concepts_terms_and_assertions(umls_dir: Path):
    neo4j = RecordingNeo4j()
    loader = Neo4jLoader(neo4j=neo4j, graph_version="kg_test", batch_size=100)
    pipeline = UMLSIngestionPipeline(
        neo4j_loader=loader,
        postgres=None,  # skip Postgres for this test
        predicate_mapper=UMLSPredicateMapper(),
        data_dir=umls_dir,
        graph_version="kg_test",
    )
    counts = pipeline.run(release="local")

    # MRCONSO: 3 ENG rows, 2 unique CUIs → 2 concepts, 3 terms
    assert counts["concepts"] >= 2
    assert counts["terms"] == 3
    # MRSTY: 2 semantic type rows
    assert counts["semantic_types"] == 2
    # MRREL: 1 relation row → 1 UMLS assertion (and 2 stub concept upserts)
    assert counts["assertions_concept"] == 1


def test_pipeline_skip_stages(umls_dir: Path):
    neo4j = RecordingNeo4j()
    loader = Neo4jLoader(neo4j=neo4j, graph_version="kg_test")
    pipeline = UMLSIngestionPipeline(
        neo4j_loader=loader,
        postgres=None,
        data_dir=umls_dir,
        graph_version="kg_test",
    )
    counts = pipeline.run(release="local", skip=["mrsty", "mrrel"])
    assert counts["semantic_types"] == 0
    assert counts["assertions_concept"] == 0
    assert counts["terms"] == 3


def test_pipeline_missing_file_raises(tmp_path: Path):
    (tmp_path / "local").mkdir()
    loader = Neo4jLoader(neo4j=RecordingNeo4j(), graph_version="kg_test")
    pipeline = UMLSIngestionPipeline(
        neo4j_loader=loader,
        postgres=None,
        data_dir=tmp_path,
    )
    with pytest.raises(FileNotFoundError, match="UMLS file not found"):
        pipeline.run(release="local")


def test_pipeline_mrrel_stubs_unknown_concepts(umls_dir: Path):
    """MRREL-only run should still create Concept stubs for CUIs it references."""
    neo4j = RecordingNeo4j()
    loader = Neo4jLoader(neo4j=neo4j, graph_version="kg_test", batch_size=100)
    pipeline = UMLSIngestionPipeline(
        neo4j_loader=loader,
        postgres=None,
        data_dir=umls_dir,
        graph_version="kg_test",
    )
    counts = pipeline.run(release="local", skip=["mrconso", "mrsty"])
    # Two unique CUIs in MRREL row 1
    assert counts["concepts"] == 2
    assert counts["assertions_concept"] == 1
