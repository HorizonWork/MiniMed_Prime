"""Pseudocode PrimeKG normalizer."""
from __future__ import annotations

from biomed_kg.common.hashing import hash_record
from biomed_kg.common.ids import make_entity_id
from biomed_kg.ingestion.base import NormalizedRecord


def normalize_primekg_node(node_id: str, name: str, node_type: str) -> dict:
    entity_type = node_type[:1].upper() + node_type[1:]
    return {"source_id": node_id, "entity_id": make_entity_id(entity_type, "primekg", node_id), "entity_type": entity_type, "preferred_name": name, "source_system": "primekg"}


def normalize_primekg_edge(record, predicate_mapper) -> list[NormalizedRecord]:
    subject = normalize_primekg_node(record.source_node_id, record.source_node_name, record.source_node_type)
    obj = normalize_primekg_node(record.target_node_id, record.target_node_name, record.target_node_type)
    predicate = predicate_mapper.map_source_predicate("primekg", record.relation, subject["entity_type"], obj["entity_type"])
    return [
        NormalizedRecord("source_entity", subject),
        NormalizedRecord("source_entity", obj),
        NormalizedRecord("source_assertion", {"subject_source_id": subject["source_id"], "predicate": predicate, "object_source_id": obj["source_id"], "source_predicate": record.relation, "source_system": "primekg", "source_record_id": hash_record(record)}),
    ]
