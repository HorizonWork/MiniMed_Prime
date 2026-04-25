"""Unit tests for the Reactome prefix and crosswalk diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from minimed_rag.kg_build.umls_crosswalk import (
    CrosswalkRunner,
    _prefix_of,
    _source_id_to_sab_code,
)


def test_reactome_human_prefix_yields_reactome_candidates():
    cands = _source_id_to_sab_code("Pathway", "R-HSA-12345")
    assert ("REACTOME", "R-HSA-12345") in cands
    assert ("REACT", "R-HSA-12345") in cands


def test_reactome_mouse_prefix_yields_reactome_candidates():
    cands = _source_id_to_sab_code("Pathway", "R-MMU-67890")
    assert ("REACTOME", "R-MMU-67890") in cands


def test_react_prefix_alternative_form():
    cands = _source_id_to_sab_code("Pathway", "REACT:42")
    assert ("REACTOME", "REACT:42") in cands


def test_existing_mondo_still_works():
    cands = _source_id_to_sab_code("Disease", "MONDO:0007256")
    assert cands == [("MONDO", "MONDO:0007256")]


def test_prefix_of_recognises_known_prefixes():
    assert _prefix_of("MONDO:1") == "MONDO"
    assert _prefix_of("R-HSA-100") == "R-HSA"
    assert _prefix_of("DB00001") == "DB"
    assert _prefix_of("12345") == "(numeric)"
    assert _prefix_of("") == "(empty)"
    assert _prefix_of("WEIRD:X") == "(other)"


def test_mesh_prefix_yields_msh_candidates():
    cands = _source_id_to_sab_code("Disease", "MESH:D012559")
    assert ("MSH", "MESH:D012559") in cands
    assert ("MSH", "D012559") in cands  # bare-code fallback


def test_msh_short_prefix_works_too():
    cands = _source_id_to_sab_code("Chemical", "MSH:D000001")
    assert ("MSH", "MSH:D000001") in cands
    assert ("MSH", "D000001") in cands


def test_ctd_prefix_yields_ctd_and_msh_fallback():
    cands = _source_id_to_sab_code("Chemical", "CTD:D012345")
    assert ("CTD", "CTD:D012345") in cands
    # CTD ids are MeSH-based; MSH fallback should be present.
    assert ("MSH", "D012345") in cands


def test_ensembl_gene_yields_ensembl_candidate():
    cands = _source_id_to_sab_code("Gene", "ENSG00000123456")
    assert ("ENSEMBL", "ENSG00000123456") in cands
    # HGNC fallback is included for cross-ref UMLS loads.
    assert ("HGNC", "ENSG00000123456") in cands


def test_ensembl_protein_yields_ensembl_candidate():
    cands = _source_id_to_sab_code("Protein", "ENSP00000123456")
    assert ("ENSEMBL", "ENSP00000123456") in cands


def test_ensembl_for_non_gene_entity_yields_no_candidates():
    """Ensembl IDs only mean something for Gene/Protein-shaped entities."""
    cands = _source_id_to_sab_code("Disease", "ENSG00000999999")
    assert cands == []


def test_prefix_of_recognises_new_prefixes():
    assert _prefix_of("MESH:D012559") == "MESH"
    assert _prefix_of("MSH:D000001") == "MSH"
    assert _prefix_of("CTD:D012345") == "CTD"
    assert _prefix_of("ENSG00000123456") == "ENSG"
    assert _prefix_of("ENST00000123456") == "ENST"
    assert _prefix_of("ENSP00000123456") == "ENSP"


@dataclass
class _FakeNeo4j:
    rows: list[dict]
    written: list = None

    def __post_init__(self):
        self.written = []

    def query(self, cypher: str, params: dict):
        return list(self.rows)

    def run(self, cypher: str, **params):
        self.written.append((cypher, params))


@dataclass
class _FakePostgres:
    cui_for: dict[tuple[str, str], str]

    def execute(self, sql: str, params: dict) -> list[tuple[str]]:
        cui = self.cui_for.get((params["sab"], params["code"]))
        return [(cui,)] if cui else []


def test_run_records_per_prefix_match_rate():
    neo4j = _FakeNeo4j(rows=[
        {"entity_id": "E1", "entity_type": "Disease", "source_id": "MONDO:1"},
        {"entity_id": "E2", "entity_type": "Disease", "source_id": "MONDO:2"},
        {"entity_id": "E3", "entity_type": "Pathway", "source_id": "R-HSA-100"},
    ])
    pg = _FakePostgres(cui_for={
        ("MONDO", "MONDO:1"): "C001",
        ("REACTOME", "R-HSA-100"): "C100",
    })
    runner = CrosswalkRunner(neo4j=neo4j, postgres=pg)
    counts = runner.run()

    assert counts["examined"] == 3
    assert counts["matched"] == 2  # MONDO:1 + R-HSA-100 matched, MONDO:2 missed
    assert "MONDO" in runner.per_prefix
    assert runner.per_prefix["MONDO"] == {"examined": 2, "matched": 1}
    assert runner.per_prefix["R-HSA"] == {"examined": 1, "matched": 1}


def test_prefix_match_rate_summary_renders_table():
    runner = CrosswalkRunner(neo4j=None, postgres=None)
    runner.per_prefix = {
        "MONDO": {"examined": 100, "matched": 80},
        "DB": {"examined": 50, "matched": 25},
    }
    summary = runner.prefix_match_rate_summary()
    assert "MONDO" in summary
    assert "80.0%" in summary
    assert "DB" in summary
    assert "50.0%" in summary
