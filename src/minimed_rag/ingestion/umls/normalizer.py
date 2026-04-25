"""UMLS normalizers — raw RRF records → canonical Neo4j / Postgres records."""

from __future__ import annotations

from minimed_rag.common.hashing import hash_text
from minimed_rag.common.ids import make_assertion_id, make_concept_id
from minimed_rag.ingestion.base import NormalizedRecord, RawRecord
from minimed_rag.ingestion.umls.predicate_map import UMLSPredicateMapper


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def _make_term_id(aui: str) -> str:
    return f"UMLS:AUI:{aui}"


def mrconso_row_to_records(
    row: dict,
    release: str = "local",
    graph_version: str = "kg_local",
) -> list[NormalizedRecord]:
    """Emit a Concept stub + a Term record + a crosswalk row per MRCONSO row."""
    cui = row["cui"]
    concept = NormalizedRecord(
        "concept",
        {
            "concept_id": make_concept_id(cui),
            "cui": cui,
            "preferred_name": row["str"] if row.get("is_preferred") else None,
            "graph_version": graph_version,
        },
    )
    term = NormalizedRecord(
        "term",
        {
            "term_id": _make_term_id(row["aui"]),
            "cui": cui,
            "concept_id": make_concept_id(cui),
            "text": row["str"],
            "normalized_text": _normalize_text(row["str"]),
            "language": row["lat"],
            "source_system": row["sab"],
            "is_preferred": bool(row.get("is_preferred", False)),
        },
    )
    crosswalk = NormalizedRecord(
        "umls_crosswalk",
        {
            "cui": cui,
            "sab": row["sab"],
            "code": row["code"],
        },
    )
    _ = release  # reserved for future per-release partitioning
    return [concept, term, crosswalk]


def mrsty_row_to_records(row: dict) -> list[NormalizedRecord]:
    return [
        NormalizedRecord(
            "concept_semantic_type",
            {
                "cui": row["cui"],
                "tui": row["tui"],
                "sty": row["sty"],
            },
        )
    ]


def mrrel_row_to_records(
    row: dict,
    mapper: UMLSPredicateMapper,
    release: str = "local",
    graph_version: str = "kg_local",
) -> list[NormalizedRecord]:
    mapping = mapper.map(row["rel"], row.get("rela") or None)
    source_record_id = row.get("rui") or hash_text(
        f"{row['cui1']}|{row['rel']}|{row.get('rela', '')}|{row['cui2']}"
    )
    subject_cui = row["cui1"]
    object_cui = row["cui2"]
    subject_id = make_concept_id(subject_cui)
    object_id = make_concept_id(object_cui)
    predicate = mapping.predicate
    assertion_id = make_assertion_id(subject_id, predicate, object_id, "umls", source_record_id)
    return [
        NormalizedRecord(
            "source_assertion",
            {
                "assertion_id": assertion_id,
                "subject_cui": subject_cui,
                "object_cui": object_cui,
                "predicate": predicate,
                "polarity": mapping.polarity,
                "confidence": mapping.confidence,
                "source_system": "umls",
                "source_predicate": (
                    f"{row['rel']}:{row['rela']}" if row.get("rela") else row["rel"]
                ),
                "source_record_id": source_record_id,
                "source_release": release,
                "graph_version": graph_version,
                "is_current": True,
            },
        )
    ]


# ── Legacy adapters for the generic SourceIngestionPipeline flow ────────────
def normalize_umls_term(record: RawRecord) -> list[NormalizedRecord]:
    return mrconso_row_to_records(record.payload)


def normalize_umls_semantic_type(record: RawRecord) -> list[NormalizedRecord]:
    return mrsty_row_to_records(record.payload)


def normalize_umls_relation(record: RawRecord) -> list[NormalizedRecord]:
    return mrrel_row_to_records(record.payload, UMLSPredicateMapper())


def normalize_umls_hierarchy(record: RawRecord) -> list[NormalizedRecord]:
    return [NormalizedRecord("hierarchy", record.payload, {"source": "umls"})]


def normalize_umls_definition(record: RawRecord) -> list[NormalizedRecord]:
    return [NormalizedRecord("definition", record.payload, {"source": "umls"})]


def normalize_umls_mapping(record: RawRecord) -> list[NormalizedRecord]:
    return [NormalizedRecord("mapping", record.payload, {"source": "umls"})]
