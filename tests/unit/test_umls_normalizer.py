"""Unit tests for UMLS normalizers and predicate mapping."""

from __future__ import annotations

from minimed_rag.ingestion.umls.normalizer import (
    _make_term_id,
    _normalize_text,
    mrconso_row_to_records,
    mrrel_row_to_records,
    mrsty_row_to_records,
)
from minimed_rag.ingestion.umls.predicate_map import (
    DEFAULT_MAPPING,
    UMLSPredicateMapper,
)


def test_normalize_text_lowercases_and_collapses_whitespace():
    assert _normalize_text("  Type 2   Diabetes  ") == "type 2 diabetes"


def test_make_term_id_uses_aui():
    assert _make_term_id("A1234").startswith("UMLS:AUI:")
    assert _make_term_id("A1234") == "UMLS:AUI:A1234"


def test_mrconso_row_to_records_shape():
    row = {
        "cui": "C0025598",
        "aui": "A0001",
        "sab": "DRUGBANK",
        "tty": "IN",
        "code": "DB00331",
        "str": "Metformin",
        "lat": "ENG",
        "is_preferred": True,
    }
    records = mrconso_row_to_records(row, release="local", graph_version="kg_test")
    types = [r.record_type for r in records]
    assert types == ["concept", "term", "umls_crosswalk"]

    concept = records[0].payload
    assert concept["cui"] == "C0025598"
    assert concept["concept_id"] == "UMLS:C0025598"
    assert concept["preferred_name"] == "Metformin"

    term = records[1].payload
    assert term["term_id"] == "UMLS:AUI:A0001"
    assert term["normalized_text"] == "metformin"
    assert term["is_preferred"] is True
    assert term["source_system"] == "DRUGBANK"

    crosswalk = records[2].payload
    assert crosswalk == {"cui": "C0025598", "sab": "DRUGBANK", "code": "DB00331"}


def test_mrconso_row_leaves_preferred_null_when_not_preferred():
    row = {
        "cui": "C1",
        "aui": "A1",
        "sab": "MSH",
        "tty": "SY",
        "code": "X",
        "str": "synonym",
        "lat": "ENG",
        "is_preferred": False,
    }
    records = mrconso_row_to_records(row)
    assert records[0].payload["preferred_name"] is None


def test_mrsty_row_to_records():
    row = {"cui": "C0025598", "tui": "T121", "sty": "Pharmacologic Substance"}
    records = mrsty_row_to_records(row)
    assert len(records) == 1
    assert records[0].record_type == "concept_semantic_type"
    assert records[0].payload["tui"] == "T121"


def test_umls_predicate_mapper_may_treat_becomes_treats():
    mapper = UMLSPredicateMapper()
    m = mapper.map("RO", "may_treat")
    assert m.predicate == "treats"
    assert m.confidence == 0.75
    assert m.polarity == "positive"


def test_umls_predicate_mapper_contraindication_is_negative():
    mapper = UMLSPredicateMapper()
    m = mapper.map("RO", "has_contraindication")
    assert m.predicate == "treats"
    assert m.polarity == "negative"


def test_umls_predicate_mapper_rel_par_falls_back_to_associated_with():
    mapper = UMLSPredicateMapper()
    m = mapper.map("PAR", "")
    assert m.predicate == "associated_with"
    assert m.confidence == 0.80


def test_umls_predicate_mapper_unknown_returns_default():
    mapper = UMLSPredicateMapper()
    m = mapper.map("UNKNOWN", "")
    assert m == DEFAULT_MAPPING


def test_mrrel_row_to_records_produces_assertion():
    row = {
        "cui1": "C0025598",
        "cui2": "C0011849",
        "rel": "RO",
        "rela": "may_treat",
        "rui": "R001",
        "sab": "DRUGBANK",
    }
    records = mrrel_row_to_records(row, UMLSPredicateMapper(), graph_version="kg_test")
    assert len(records) == 1
    assertion = records[0].payload
    assert assertion["predicate"] == "treats"
    assert assertion["confidence"] == 0.75
    assert assertion["subject_cui"] == "C0025598"
    assert assertion["object_cui"] == "C0011849"
    assert assertion["source_predicate"] == "RO:may_treat"
    assert assertion["source_system"] == "umls"
    assert assertion["assertion_id"].startswith("ASSERT:")
    assert assertion["is_current"] is True
