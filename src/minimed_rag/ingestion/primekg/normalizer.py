"""PrimeKG normalizer — raw CSV rows → canonical entity/assertion records.

PrimeKG's CSV is undirected by construction: each edge has an x_id/y_id
ordering that comes from the source dump, not from the predicate's
intended directionality. For example ``pathway_protein`` rows have
x=Pathway, y=Protein, but the canonical predicate ``participates_in``
declares Gene/Protein → Pathway. Storing the assertion as
``(Pathway, participates_in, Protein)`` would force every downstream
consumer (path serialization, TRM training) to re-derive direction.

Phase 6.5 fix: when a ``PredicateRegistry`` is supplied, the normalizer
checks each directional assertion. If subject_type mismatches the
predicate's domain but matches its range (and vice-versa for object),
the endpoints are swapped so the stored assertion is canonical. The
query layer (PathFinder._resolve_predicate) still works either way; the
swap eliminates a class of subtle bugs in path serialization for Phase 7.
"""

from __future__ import annotations

from typing import Any

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


def _should_swap_for_canonical_direction(
    subject_type: str,
    object_type: str,
    predicate: str,
    predicate_registry: Any,
) -> bool:
    """Return True if (subject, object) should be swapped to align with
    the predicate's declared (domain, range).

    Returns False — i.e., keep PrimeKG order — when:
    - registry is None (caller opted out),
    - the predicate isn't declared (unknown predicate fallback),
    - the predicate is non-directional (swap is meaningless),
    - PrimeKG order already matches domain/range,
    - PrimeKG order matches NEITHER direction (don't pick one arbitrarily).
    """
    if predicate_registry is None:
        return False
    try:
        rule = predicate_registry.get(predicate)
    except KeyError:
        return False
    if not rule.directional:
        return False
    forward = predicate_registry.matches_type(
        subject_type, rule.domain
    ) and predicate_registry.matches_type(object_type, rule.range)
    if forward:
        return False
    inverse = predicate_registry.matches_type(
        subject_type, rule.range
    ) and predicate_registry.matches_type(object_type, rule.domain)
    return inverse


def normalize_primekg_edge(
    record,
    predicate_mapper,
    source_release: str = "local",
    graph_version: str = "kg_local",
    predicate_registry: Any = None,
) -> list[NormalizedRecord]:
    """Produce one Entity record per endpoint and one reified Assertion record.

    The mapper is expected to return a ``PredicateMapping`` (predicate,
    confidence, polarity); a bare string is also accepted for back-compat.

    When ``predicate_registry`` is supplied, directional predicates whose
    PrimeKG row order is reversed relative to the predicate's declared
    domain/range get their (subject, object) swapped. This makes the
    stored assertion canonical and removes ambiguity for downstream
    path serialization (Phase 7 TRM training).
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

    if _should_swap_for_canonical_direction(
        subject["entity_type"], obj["entity_type"], predicate, predicate_registry
    ):
        subject, obj = obj, subject

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
