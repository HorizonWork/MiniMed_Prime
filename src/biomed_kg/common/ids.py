"""Pseudocode for stable deterministic IDs."""
from __future__ import annotations

import hashlib


def stable_hash(*parts: object, length: int = 16) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def make_entity_id(entity_type: str, source_system: str, source_id: str) -> str:
    return f"KG:{entity_type}:{stable_hash(source_system, source_id)}"


def make_concept_id(cui: str) -> str:
    return f"UMLS:{cui}"


def make_assertion_id(subject_id: str, predicate: str, object_id: str, source_system: str, source_record_id: str) -> str:
    return f"ASSERT:{stable_hash(subject_id, predicate, object_id, source_system, source_record_id)}"


def make_path_id(task_id: str, metapath_template_id: str, assertion_ids: list[str]) -> str:
    return f"PATH:{stable_hash(task_id, metapath_template_id, *assertion_ids)}"


def make_vector_id(object_type: str, object_id: str) -> str:
    return f"VEC:{object_type}:{stable_hash(object_id)}"


def make_section_id(doc_id: str, section_title: str) -> str:
    return f"SEC:{stable_hash(doc_id, section_title)}"
