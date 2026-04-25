"""Phase 4 end-to-end integration test.

Gated by ``RUN_INTEGRATION=1`` and a reachable Neo4j/Postgres. Loads a tiny
synthetic PrimeKG CSV, applies the schema, runs the CLI ingest, then queries
``GraphClient.get_neighbors`` and asserts expected neighbors appear.

Do **not** run as part of the default unit suite — it requires live
containers (`make up`) and bootstrapped Postgres (`bash
scripts/bootstrap_postgres_schema.sh`).
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="set RUN_INTEGRATION=1 to run integration tests",
    ),
]


@pytest.fixture(scope="module")
def primekg_csv(tmp_path_factory) -> Path:
    csv = tmp_path_factory.mktemp("primekg_e2e") / "kg.csv"
    csv.write_text(
        textwrap.dedent(
            """\
            x_id,x_name,x_type,relation,y_id,y_name,y_type,source
            DB00331,Metformin,drug,indication,MONDO:0005148,Type 2 diabetes mellitus,disease,curated
            DB00331,Metformin,drug,drug_protein,5562,AMPK,gene/protein,curated
            MONDO:0005148,Type 2 diabetes mellitus,disease,disease_protein,3630,INS,gene/protein,curated
            """
        ),
        encoding="utf-8",
    )
    return csv


def test_primekg_ingest_and_get_neighbors(primekg_csv: Path) -> None:
    pytest.importorskip("neo4j")
    from minimed_rag.common.config import get_settings
    from minimed_rag.ingestion.primekg.pipeline import PrimeKGIngestionPipeline
    from minimed_rag.ingestion.primekg.predicate_map import SimplePredicateMapper
    from minimed_rag.kg.graph_client import GraphClient
    from minimed_rag.kg_build.neo4j_loader import Neo4jLoader
    from minimed_rag.storage.neo4j import connect as neo4j_connect

    class StaticDownloader:
        def ensure_download(self, release: str) -> Path:
            return primekg_csv

    settings = get_settings()
    neo4j = neo4j_connect(
        settings.neo4j_uri,
        settings.neo4j_user,
        settings.neo4j_password,
        settings.neo4j_database,
    )
    try:
        # Clear any prior run data for idempotency
        neo4j.run(
            "MATCH (a:Assertion) WHERE a.graph_version = $gv DETACH DELETE a",
            gv="kg_e2e",
        )
        neo4j.run(
            "MATCH (e:Entity) WHERE e.graph_version = $gv DETACH DELETE e",
            gv="kg_e2e",
        )

        loader = Neo4jLoader(neo4j, graph_version="kg_e2e", batch_size=100)
        loader.apply_schema()

        pipeline = PrimeKGIngestionPipeline(
            downloader=StaticDownloader(),
            predicate_mapper=SimplePredicateMapper(),
            neo4j_loader=loader,
            graph_version="kg_e2e",
        )
        counts = pipeline.run(release="local")
        assert counts["entities"] == 4  # Metformin + T2DM + AMPK + INS
        assert counts["assertions_entity"] == 3

        graph = GraphClient(neo4j=neo4j)
        # Neighbors of metformin should include the disease and the protein
        metformin_rows = neo4j.query(
            "MATCH (e:Entity {source_system: 'primekg', source_id: 'DB00331'}) "
            "RETURN e.entity_id AS id",
            {},
        )
        assert metformin_rows
        metformin_id = metformin_rows[0]["id"]

        neighbors = graph.get_neighbors(metformin_id, depth=1, min_confidence=0.0)
        neighbor_preds = {n.predicate for n in neighbors}
        assert "treats" in neighbor_preds
        assert "targets" in neighbor_preds
    finally:
        neo4j.close()
