"""Round-trip and integrity tests for config/kg_schema.yaml."""

from __future__ import annotations

import pytest

from kg.schema import ANY_NODE, KGSchema, load_schema


@pytest.fixture(scope="module")
def schema() -> KGSchema:
    return load_schema()


def test_loads_and_has_expected_counts(schema: KGSchema) -> None:
    assert len(schema.node_types) == 12
    assert len(schema.edge_types) == 40
    assert schema.meta.primekg_inherited == 30
    assert schema.meta.clinical_extension == 10


def test_all_12_node_types_present(schema: KGSchema) -> None:
    expected = {
        "gene_protein", "drug", "disease", "phenotype", "anatomy",
        "biological_process", "molecular_function", "cellular_component",
        "pathway", "exposure", "symptom", "clinical_object",
    }
    assert set(schema.node_types) == expected


def test_clinical_object_has_subtypes(schema: KGSchema) -> None:
    co = schema.node_types["clinical_object"]
    assert co.subtypes == ["lab_test", "procedure", "clinical_guideline"]


def test_every_node_declares_at_least_one_id(schema: KGSchema) -> None:
    for name, node in schema.node_types.items():
        assert node.ids, f"node {name} has no ids"


def test_every_edge_endpoint_resolves(schema: KGSchema) -> None:
    valid = set(schema.node_types) | {ANY_NODE}
    for name, edge in schema.edge_types.items():
        assert edge.head_type in valid, f"{name}: head_type {edge.head_type} unknown"
        assert edge.tail_type in valid, f"{name}: tail_type {edge.tail_type} unknown"


def test_is_a_edges_marked_dag(schema: KGSchema) -> None:
    parent_edges = [n for n in schema.edge_types if n.endswith("_PARENT")]
    assert parent_edges, "expected at least one *_PARENT edge"
    for n in parent_edges:
        assert schema.edge_types[n].dag is True, f"{n} should have dag=true"


def test_known_symmetric_edges(schema: KGSchema) -> None:
    for n in ("PROTEIN_PROTEIN", "DRUG_DRUG_INTERACTION", "XREF_CUI"):
        assert schema.edge_types[n].symmetric is True, f"{n} should be symmetric"


def test_directed_edges_are_not_symmetric(schema: KGSchema) -> None:
    for n in ("DRUG_INDICATION", "DISEASE_GENE_ASSOC", "GUIDELINE_DRUG"):
        assert schema.edge_types[n].symmetric is False


def test_xref_cui_uses_any_endpoints(schema: KGSchema) -> None:
    xref = schema.edge_types["XREF_CUI"]
    assert xref.head_type == ANY_NODE
    assert xref.tail_type == ANY_NODE
    assert "cui" in xref.props


def test_edge_global_props_present(schema: KGSchema) -> None:
    assert set(schema.edge_global_props) >= {
        "source", "evidence_level", "n_pubmed_citations",
        "last_curated_date", "weight",
    }


def test_meta_count_invariants(schema: KGSchema) -> None:
    assert schema.meta.total_node_types == len(schema.node_types)
    assert schema.meta.total_edge_types == len(schema.edge_types)
    assert (
        schema.meta.primekg_inherited + schema.meta.clinical_extension
        == schema.meta.total_edge_types
    )
