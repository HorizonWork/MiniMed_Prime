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


# ---- Phase 6.5 direction-canonicalization tests --------------------------

# PrimeKG is undirected by construction; the x_id/y_id ordering comes from
# the source dump, not from the predicate's intended directionality. With a
# PredicateRegistry, the normalizer should swap (subject, object) when
# PrimeKG order is reversed relative to the predicate's domain/range.

from minimed_rag.schema_registry.predicate_registry import PredicateRegistry


def _registry() -> PredicateRegistry:
    return PredicateRegistry()


def test_pathway_protein_swaps_to_protein_pathway():
    """PrimeKG ``pathway_protein`` row maps to ``participates_in`` whose
    domain is Gene/Protein and range is Pathway. The row has x=Pathway,
    y=Protein; without the swap we'd store backwards."""
    edge = RawPrimeKGEdge(
        source_node_id="REACTOME:R-HSA-1",
        source_node_name="Glycolysis",
        source_node_type="pathway",
        relation="pathway_protein",
        target_node_id="9606",
        target_node_name="HK1",
        target_node_type="gene/protein",
    )
    assertion = normalize_primekg_edge(
        edge, SimplePredicateMapper(), predicate_registry=_registry()
    )[2].payload
    assert assertion["predicate"] == "participates_in"
    assert assertion["subject_id"].startswith("KG:GeneProtein:")
    assert assertion["object_id"].startswith("KG:Pathway:")


def test_drug_protein_keeps_drug_subject():
    """PrimeKG ``drug_protein`` row maps to ``targets`` (Drug→Protein).
    Already canonical; no swap should fire."""
    edge = RawPrimeKGEdge(
        source_node_id="DrugBank:DB00001",
        source_node_name="Lepirudin",
        source_node_type="drug",
        relation="drug_protein",
        target_node_id="9606",
        target_node_name="F2",
        target_node_type="gene/protein",
    )
    assertion = normalize_primekg_edge(
        edge, SimplePredicateMapper(), predicate_registry=_registry()
    )[2].payload
    assert assertion["predicate"] == "targets"
    assert assertion["subject_id"].startswith("KG:Drug:")
    assert assertion["object_id"].startswith("KG:GeneProtein:")


def test_indication_keeps_drug_disease_order():
    """``treats`` is Drug→Disease and PrimeKG already uses that order."""
    edge = RawPrimeKGEdge(
        source_node_id="DrugBank:DB1",
        source_node_name="Metformin",
        source_node_type="drug",
        relation="indication",
        target_node_id="MONDO:1",
        target_node_name="T2D",
        target_node_type="disease",
    )
    assertion = normalize_primekg_edge(
        edge, SimplePredicateMapper(), predicate_registry=_registry()
    )[2].payload
    assert assertion["subject_id"].startswith("KG:Drug:")
    assert assertion["object_id"].startswith("KG:Disease:")


def test_non_directional_predicate_never_swaps():
    """``associated_with`` is non-directional; even if PrimeKG order is
    'wrong' biomedically, the swap is meaningless and must not fire."""
    edge = RawPrimeKGEdge(
        source_node_id="MONDO:1",
        source_node_name="DiseaseA",
        source_node_type="disease",
        relation="disease_disease",
        target_node_id="MONDO:2",
        target_node_name="DiseaseB",
        target_node_type="disease",
    )
    records = normalize_primekg_edge(
        edge, SimplePredicateMapper(), predicate_registry=_registry()
    )
    subject_payload = records[0].payload
    object_payload = records[1].payload
    assertion = records[2].payload
    assert assertion["predicate"] == "associated_with"
    # PrimeKG row order preserved (first source_entity is the subject).
    assert subject_payload["source_id"] == "MONDO:1"
    assert object_payload["source_id"] == "MONDO:2"
    assert assertion["subject_id"] == subject_payload["entity_id"]
    assert assertion["object_id"] == object_payload["entity_id"]


def test_unknown_predicate_does_not_crash_with_registry_kept_order():
    """Restated for clarity using payload-level checks (not source_id
    suffix matching)."""
    edge = RawPrimeKGEdge(
        "x:1", "X", "drug", "unknown_relation_xyz", "y:1", "Y", "disease"
    )
    records = normalize_primekg_edge(
        edge, SimplePredicateMapper(), predicate_registry=_registry()
    )
    assert records[0].payload["source_id"] == "x:1"
    assert records[1].payload["source_id"] == "y:1"
    assert records[2].payload["predicate"] == "associated_with"


def test_no_registry_keeps_legacy_behavior():
    """Without a registry, the normalizer behaves as Phase 4: no swap.
    Backward-compat for any caller that doesn't yet inject the registry."""
    edge = RawPrimeKGEdge(
        source_node_id="REACTOME:R-HSA-1",
        source_node_name="Glycolysis",
        source_node_type="pathway",
        relation="pathway_protein",
        target_node_id="9606",
        target_node_name="HK1",
        target_node_type="gene/protein",
    )
    assertion = normalize_primekg_edge(edge, SimplePredicateMapper())[2].payload
    # No registry → PrimeKG order preserved.
    assert assertion["subject_id"].startswith("KG:Pathway:")
    assert assertion["object_id"].startswith("KG:GeneProtein:")


def test_assertion_id_changes_when_swap_fires():
    """Sanity: if endpoints are swapped, the deterministic assertion_id
    should reflect the new (subject, object) order. Otherwise downstream
    dedup against the legacy IDs would silently fail."""
    edge = RawPrimeKGEdge(
        "REACTOME:R-HSA-1", "Glycolysis", "pathway",
        "pathway_protein",
        "9606", "HK1", "gene/protein",
    )
    no_reg = normalize_primekg_edge(edge, SimplePredicateMapper())[2].payload
    with_reg = normalize_primekg_edge(
        edge, SimplePredicateMapper(), predicate_registry=_registry()
    )[2].payload
    assert no_reg["assertion_id"] != with_reg["assertion_id"]
