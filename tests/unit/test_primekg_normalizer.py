"""Unit tests for the PrimeKG normalizer."""

from __future__ import annotations

from minimed_rag.ingestion.primekg.normalizer import (
    normalize_entity_type,
    normalize_primekg_edge,
    normalize_primekg_node,
)
from minimed_rag.ingestion.primekg.parser import RawPrimeKGEdge
from minimed_rag.ingestion.primekg.predicate_map import SimplePredicateMapper


def test_normalize_entity_type_handles_slash_and_underscore():
    assert normalize_entity_type("gene/protein") == "GeneProtein"
    assert normalize_entity_type("biological_process") == "BiologicalProcess"
    assert normalize_entity_type("drug") == "Drug"
    assert normalize_entity_type("disease") == "Disease"
    assert normalize_entity_type("molecular_function") == "MolecularFunction"


def test_normalize_primekg_node_shape():
    node = normalize_primekg_node(
        "MONDO:0005015",
        "Type 2 diabetes mellitus",
        "disease",
        source_release="local",
        graph_version="kg_test",
    )
    assert node["source_id"] == "MONDO:0005015"
    assert node["entity_type"] == "Disease"
    assert node["preferred_name"] == "Type 2 diabetes mellitus"
    assert node["source_system"] == "primekg"
    assert node["graph_version"] == "kg_test"
    assert node["canonical_cui"] is None
    assert node["entity_id"].startswith("KG:Disease:")


def test_normalize_primekg_edge_indication_produces_treats_with_confidence():
    edge = RawPrimeKGEdge(
        source_node_id="DrugBank:0001",
        source_node_name="Metformin",
        source_node_type="drug",
        relation="indication",
        target_node_id="MONDO:0005015",
        target_node_name="Type 2 diabetes mellitus",
        target_node_type="disease",
    )
    records = normalize_primekg_edge(edge, SimplePredicateMapper(), graph_version="kg_test")

    assert [r.record_type for r in records] == [
        "source_entity",
        "source_entity",
        "source_assertion",
    ]
    assertion = records[2].payload
    assert assertion["predicate"] == "treats"
    assert assertion["confidence"] == 0.90
    assert assertion["polarity"] == "positive"
    assert assertion["source_system"] == "primekg"
    assert assertion["source_predicate"] == "indication"
    assert assertion["subject_id"].startswith("KG:Drug:")
    assert assertion["object_id"].startswith("KG:Disease:")
    assert assertion["assertion_id"].startswith("ASSERT:")
    assert assertion["graph_version"] == "kg_test"
    assert assertion["is_current"] is True


def test_normalize_primekg_edge_contraindication_flips_polarity():
    edge = RawPrimeKGEdge(
        "DrugBank:1",
        "Aspirin",
        "drug",
        "contraindication",
        "MONDO:2",
        "Gastric ulcer",
        "disease",
    )
    assertion = normalize_primekg_edge(edge, SimplePredicateMapper())[2].payload
    assert assertion["predicate"] == "treats"
    assert assertion["polarity"] == "negative"
    assert assertion["confidence"] == 0.85


def test_normalize_primekg_edge_unknown_relation_falls_back_to_associated_with():
    edge = RawPrimeKGEdge(
        "x:1",
        "X",
        "drug",
        "unknown_relation_xyz",
        "y:1",
        "Y",
        "disease",
    )
    assertion = normalize_primekg_edge(edge, SimplePredicateMapper())[2].payload
    assert assertion["predicate"] == "associated_with"
    assert assertion["confidence"] == 0.40


def test_normalize_primekg_edge_accepts_string_return_from_legacy_mapper():
    class LegacyMapper:
        def map_source_predicate(self, *args, **kwargs):
            return "treats"

    edge = RawPrimeKGEdge("x:1", "X", "drug", "indication", "y:1", "Y", "disease")
    assertion = normalize_primekg_edge(edge, LegacyMapper())[2].payload
    assert assertion["predicate"] == "treats"
    assert assertion["confidence"] == 0.5  # default when mapper returns bare string
    assert assertion["polarity"] == "positive"


def test_normalize_primekg_edge_deterministic_entity_and_assertion_ids():
    edge = RawPrimeKGEdge(
        "DrugBank:1",
        "Metformin",
        "drug",
        "indication",
        "MONDO:1",
        "Diabetes",
        "disease",
    )
    mapper = SimplePredicateMapper()
    records_a = normalize_primekg_edge(edge, mapper)
    records_b = normalize_primekg_edge(edge, mapper)
    assert records_a[2].payload["assertion_id"] == records_b[2].payload["assertion_id"]
    assert records_a[0].payload["entity_id"] == records_b[0].payload["entity_id"]
