"""PrimeKG normalizer — raw CSV rows → canonical entity/assertion records."""

from __future__ import annotations

from minimed_rag.common.hashing import hash_record
from minimed_rag.common.ids import make_assertion_id, make_entity_id
from minimed_rag.ingestion.base import NormalizedRecord


def normalize_entity_type(raw_type: str) -> str:
    """PrimeKG types → canonical CamelCase.

    Handles ``gene/protein`` → ``GeneProtein``,
    ``biological_process`` → ``BiologicalProcess``, etc.
    """
    cleaned = raw_type.replace("/", " ").replace("_", " ").strip()
    return "".join(part[:1].upper() + part[1:].lower() for part in cleaned.split())


def normalize_primekg_node(
    node_id: str,
    name: str,
    node_type: str,
    source_release: str = "local",
    graph_version: str = "kg_local",
) -> dict:
    entity_type = normalize_entity_type(node_type)
    return {
        "source_id": node_id,
        "entity_id": make_entity_id(entity_type, "primekg", node_id),
        "entity_type": entity_type,
        "preferred_name": name,
        "source_system": "primekg",
        "source_release": source_release,
        "graph_version": graph_version,
        "is_active": True,
        "canonical_cui": None,
    }


def normalize_primekg_edge(
    record,
    predicate_mapper,
    source_release: str = "local",
    graph_version: str = "kg_local",
) -> list[NormalizedRecord]:
    """Produce one Entity record per endpoint and one reified Assertion record.

    The mapper is expected to return a ``PredicateMapping`` (predicate,
    confidence, polarity); a bare string is also accepted for back-compat.
    """
    subject = normalize_primekg_node(
        record.source_node_id,
        record.source_node_name,
        record.source_node_type,
        source_release=source_release,
        graph_version=graph_version,
    )
    obj = normalize_primekg_node(
        record.target_node_id,
        record.target_node_name,
        record.target_node_type,
        source_release=source_release,
        graph_version=graph_version,
    )
    mapping = predicate_mapper.map_source_predicate(
        "primekg", record.relation, subject["entity_type"], obj["entity_type"]
    )
    if isinstance(mapping, str):
        predicate, confidence, polarity = mapping, 0.5, "positive"
    else:
        predicate = mapping.predicate
        confidence = mapping.confidence
        polarity = mapping.polarity

    source_record_id = hash_record(record)
    assertion_id = make_assertion_id(
        subject["entity_id"],
        predicate,
        obj["entity_id"],
        "primekg",
        source_record_id,
    )
    return [
        NormalizedRecord("source_entity", subject),
        NormalizedRecord("source_entity", obj),
        NormalizedRecord(
            "source_assertion",
            {
                "assertion_id": assertion_id,
                "subject_id": subject["entity_id"],
                "predicate": predicate,
                "object_id": obj["entity_id"],
                "polarity": polarity,
                "confidence": confidence,
                "source_system": "primekg",
                "source_predicate": record.relation,
                "source_record_id": source_record_id,
                "source_release": source_release,
                "graph_version": graph_version,
                "is_current": True,
            },
        ),
    ]
