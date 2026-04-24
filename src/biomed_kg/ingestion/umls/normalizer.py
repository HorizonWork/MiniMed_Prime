"""Pseudocode UMLS normalized record builders."""
from __future__ import annotations

from biomed_kg.ingestion.base import NormalizedRecord, RawRecord


def normalize_umls_term(record: RawRecord) -> list[NormalizedRecord]:
    return [NormalizedRecord("term", record.payload, {"source": "umls"})]


def normalize_umls_semantic_type(record: RawRecord) -> list[NormalizedRecord]:
    return [NormalizedRecord("semantic_type", record.payload, {"source": "umls"})]


def normalize_umls_relation(record: RawRecord) -> list[NormalizedRecord]:
    return [NormalizedRecord("source_assertion", record.payload, {"source": "umls"})]


def normalize_umls_hierarchy(record: RawRecord) -> list[NormalizedRecord]:
    return [NormalizedRecord("hierarchy", record.payload, {"source": "umls"})]


def normalize_umls_definition(record: RawRecord) -> list[NormalizedRecord]:
    return [NormalizedRecord("definition", record.payload, {"source": "umls"})]


def normalize_umls_mapping(record: RawRecord) -> list[NormalizedRecord]:
    return [NormalizedRecord("mapping", record.payload, {"source": "umls"})]
