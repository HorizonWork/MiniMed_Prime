from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.schemas import KGEdge

QUESTION_TYPE_PATTERNS: dict[str, tuple[str, ...]] = {
    "drug_interaction": ("interaction", "interact", "contraindicated", "combine", "co-administer", "coadminister"),
    "dosage": ("dose", "dosage", "mg", "mcg", "titrate", "how much"),
    "diagnosis": ("diagnosis", "diagnose", "differential", "most likely", "presents with", "likely condition"),
    "etiology": ("cause", "causes", "caused by", "etiology", "why does", "mechanism"),
    "factoid": (),
    "other": (),
}

QUESTION_TYPE_TO_RELATIONS: dict[str, frozenset[str]] = {
    "drug_interaction": frozenset(
        {
            "drug_drug",
            "drugdrug",
            "contraindication",
            "drug_protein",
            "indication",
            "off-label use",
            "drug_effect",
        }
    ),
    "dosage": frozenset(
        {
            "contraindication",
            "indication",
            "drug_protein",
            "pharmacologic_class",
            "drug_effect",
        }
    ),
    "diagnosis": frozenset(
        {
            "disease_disease",
            "disease_protein",
            "disease_symptom",
            "disease_phenotype_positive",
            "disease_phenotype_negative",
            "phenotype_phenotype",
        }
    ),
    "etiology": frozenset(
        {
            "disease_protein",
            "gene_disease",
            "anatomy_disease",
            "exposure_disease",
            "exposure_protein",
            "anatomy_protein_present",
            "anatomy_protein_absent",
        }
    ),
    "other": frozenset(
        {
            "contraindication",
            "indication",
            "drug_drug",
            "drug_protein",
            "drug_effect",
            "disease_disease",
            "disease_protein",
            "disease_phenotype_positive",
        }
    ),
}

DEFAULT_FALLBACK_RELATIONS: frozenset[str] = QUESTION_TYPE_TO_RELATIONS["other"]

QUESTION_TYPE_ALIASES = {
    "drug interaction": "drug_interaction",
    "drug-interaction": "drug_interaction",
    "drug_interaction": "drug_interaction",
    "ddi": "drug_interaction",
    "dosage": "dosage",
    "dose": "dosage",
    "diagnosis": "diagnosis",
    "dx": "diagnosis",
    "etiology": "etiology",
    "aetiology": "etiology",
    "factoid": "factoid",
    "other": "other",
}

PUZZLE_ID_TO_QUESTION_TYPE = {
    0: "drug_interaction",
    1: "diagnosis",
    2: "dosage",
    3: "etiology",
    4: "other",
}

TOKEN_SYNONYMS = {
    "gene": "protein",
    "symptom": "phenotype",
}

FAMILY_RELATION_EXPANSIONS: dict[str, frozenset[str]] = {
    "disease symptom": frozenset({"disease_phenotype_positive", "disease_phenotype_negative"}),
    "diseasesymptom": frozenset({"disease_phenotype_positive", "disease_phenotype_negative"}),
    "gene disease": frozenset({"disease_protein"}),
    "genedisease": frozenset({"disease_protein"}),
}


@dataclass(frozen=True, slots=True)
class RelationFilterResolution:
    question_type: str
    canonical_question_type: str
    requested_relations: frozenset[str]
    matched_relations: frozenset[str]
    requested_but_missing_relations: frozenset[str]
    override_source: str | None


def infer_question_type(
    record: Mapping[str, Any] | None = None,
    question: str | None = None,
    options: Any | None = None,
    fallback_label: str = "other",
) -> str:
    question_type = _question_type_from_record(record)
    if question_type is not None:
        return question_type

    option_text = _flatten_options(options if options is not None else (record or {}).get("options") or (record or {}).get("choices"))
    text_parts = [part for part in (question, option_text) if isinstance(part, str) and part.strip()]
    combined_text = "\n".join(text_parts).strip().lower()
    for question_type, patterns in QUESTION_TYPE_PATTERNS.items():
        if question_type in {"factoid", "other"}:
            continue
        if any(pattern in combined_text for pattern in patterns):
            return question_type
    return fallback_label


def canonicalize_question_type(question_type: str | None, *, for_routing: bool = True) -> str:
    normalized = _normalize_question_type_value(question_type)
    if normalized is None:
        return "other"
    if for_routing and normalized == "factoid":
        return "other"
    return normalized


def get_allowed_relations(
    question_type: str,
    available_relations: set[str] | Sequence[str],
    *,
    overrides: Mapping[str, set[str] | frozenset[str] | Sequence[str]] | None = None,
    explicit_override: set[str] | frozenset[str] | Sequence[str] | None = None,
) -> set[str]:
    resolution = resolve_relation_filter(
        question_type=question_type,
        available_relations=available_relations,
        overrides=overrides,
        explicit_override=explicit_override,
    )
    return set(resolution.matched_relations)


def filter_edges_by_question_type(
    edges: Sequence[KGEdge],
    question_type: str,
    available_relations: set[str] | Sequence[str] | None = None,
    *,
    overrides: Mapping[str, set[str] | frozenset[str] | Sequence[str]] | None = None,
    explicit_override: set[str] | frozenset[str] | Sequence[str] | None = None,
) -> list[KGEdge]:
    relation_pool = set(available_relations or {edge.relation for edge in edges})
    allowed_relations = get_allowed_relations(
        question_type=question_type,
        available_relations=relation_pool,
        overrides=overrides,
        explicit_override=explicit_override,
    )
    if not allowed_relations:
        return []
    return [edge for edge in edges if edge.relation in allowed_relations]


def summarize_relation_filter_stats(
    *,
    question_type: str,
    canonical_question_type: str | None = None,
    requested_relations: set[str] | frozenset[str] | Sequence[str] = (),
    matched_relations: set[str] | frozenset[str] | Sequence[str] = (),
    requested_but_missing_relations: set[str] | frozenset[str] | Sequence[str] = (),
    edges_before: int = 0,
    edges_after: int = 0,
    node_count_after: int = 0,
    fallback_stage: str = "none",
    override_source: str | None = None,
    use_relation_filter: bool = True,
) -> dict[str, Any]:
    canonical = canonical_question_type or canonicalize_question_type(question_type)
    return {
        "question_type": str(question_type),
        "question_type_used": str(question_type),
        "canonical_question_type": canonical,
        "requested_relations": sorted({str(item) for item in requested_relations}),
        "matched_relations": sorted({str(item) for item in matched_relations}),
        "requested_but_missing_relations": sorted({str(item) for item in requested_but_missing_relations}),
        "edges_before": int(edges_before),
        "edges_after": int(edges_after),
        "node_count_after": int(node_count_after),
        "fallback_stage": fallback_stage,
        "fallback_stage_used": fallback_stage,
        "override_source": override_source,
        "use_relation_filter": bool(use_relation_filter),
    }


def resolve_relation_filter(
    *,
    question_type: str,
    available_relations: set[str] | Sequence[str],
    overrides: Mapping[str, set[str] | frozenset[str] | Sequence[str]] | None = None,
    explicit_override: set[str] | frozenset[str] | Sequence[str] | None = None,
) -> RelationFilterResolution:
    relation_pool = {str(relation) for relation in available_relations}
    canonical_question_type = canonicalize_question_type(question_type)
    requested_relations, override_source = _requested_relations_for_question_type(
        question_type=canonical_question_type,
        overrides=overrides,
        explicit_override=explicit_override,
    )
    matched_relations: set[str] = set()
    requested_but_missing_relations: set[str] = set()
    for requested_relation in requested_relations:
        expanded_requested_relations = _expand_relation_family(requested_relation)
        relation_matches: set[str] = set()
        for expanded_relation in expanded_requested_relations:
            relation_matches.update(_match_relation_alias(expanded_relation, relation_pool))
        if relation_matches:
            matched_relations.update(relation_matches)
        else:
            requested_but_missing_relations.add(requested_relation)

    return RelationFilterResolution(
        question_type=str(question_type),
        canonical_question_type=canonical_question_type,
        requested_relations=frozenset(requested_relations),
        matched_relations=frozenset(matched_relations),
        requested_but_missing_relations=frozenset(requested_but_missing_relations),
        override_source=override_source,
    )


def _question_type_from_record(record: Mapping[str, Any] | None) -> str | None:
    if not record:
        return None
    for field_name in ("question_type", "type"):
        normalized = _normalize_question_type_value(record.get(field_name))
        if normalized is not None:
            return normalized

    for field_name in ("puzzle_id", "puzzle_identifiers"):
        question_type = _question_type_from_puzzle_value(record.get(field_name))
        if question_type is not None:
            return question_type
    return None


def _question_type_from_puzzle_value(value: Any) -> str | None:
    numeric_value = _coerce_int(value)
    if numeric_value is not None:
        return PUZZLE_ID_TO_QUESTION_TYPE.get(numeric_value)
    if hasattr(value, "tolist"):
        try:
            return _question_type_from_puzzle_value(value.tolist())
        except Exception:
            return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value:
            return None
        return _question_type_from_puzzle_value(value[0])
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None


def _requested_relations_for_question_type(
    *,
    question_type: str,
    overrides: Mapping[str, set[str] | frozenset[str] | Sequence[str]] | None,
    explicit_override: set[str] | frozenset[str] | Sequence[str] | None,
) -> tuple[set[str], str | None]:
    if explicit_override is not None:
        return _stringify_relation_values(explicit_override), "explicit_override"
    if overrides:
        override_relations = overrides.get(question_type)
        if override_relations is not None:
            return _stringify_relation_values(override_relations), "type_override"
    return set(QUESTION_TYPE_TO_RELATIONS.get(question_type, DEFAULT_FALLBACK_RELATIONS)), "question_type_default"


def _stringify_relation_values(values: set[str] | frozenset[str] | Sequence[str]) -> set[str]:
    if isinstance(values, (str, bytes)):
        return {str(values).strip()} if str(values).strip() else set()
    return {str(value).strip() for value in values if str(value).strip()}


def _expand_relation_family(relation_name: str) -> set[str]:
    normalized_with_separators = _separator_key(relation_name)
    normalized_compact = _compact_key(relation_name)
    expanded_relations = set(FAMILY_RELATION_EXPANSIONS.get(normalized_with_separators, ()))
    expanded_relations.update(FAMILY_RELATION_EXPANSIONS.get(normalized_compact, ()))
    if expanded_relations:
        return expanded_relations
    return {relation_name}


def _match_relation_alias(requested_relation: str, available_relations: set[str]) -> set[str]:
    requested_keys = _relation_alias_keys(requested_relation)
    matched_relations = set()
    for available_relation in available_relations:
        if requested_keys & _relation_alias_keys(available_relation):
            matched_relations.add(available_relation)
    return matched_relations


def _relation_alias_keys(value: str) -> set[str]:
    base_tokens = _relation_tokens(value)
    if not base_tokens:
        return set()
    token_variants = {tuple(base_tokens), tuple(TOKEN_SYNONYMS.get(token, token) for token in base_tokens)}
    keys: set[str] = set()
    for tokens in token_variants:
        keys.add(" ".join(tokens))
        keys.add("".join(tokens))
        keys.add("_".join(tokens))
        keys.add("-".join(tokens))
    return keys


def _relation_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value).lower())


def _separator_key(value: str) -> str:
    return " ".join(_relation_tokens(value))


def _compact_key(value: str) -> str:
    return "".join(_relation_tokens(value))


def _normalize_question_type_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if not lowered:
        return None
    normalized = lowered.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return QUESTION_TYPE_ALIASES.get(lowered) or QUESTION_TYPE_ALIASES.get(normalized)


def _flatten_options(options: Any) -> str:
    if options is None:
        return ""
    if isinstance(options, Mapping):
        option_lines = [f"{key}: {value}" for key, value in options.items()]
        return "Options: " + "; ".join(option_lines) if option_lines else ""
    if isinstance(options, Sequence) and not isinstance(options, (str, bytes)):
        option_lines = [str(item) for item in options if str(item).strip()]
        return "Options: " + "; ".join(option_lines) if option_lines else ""
    return str(options).strip()


__all__ = [
    "DEFAULT_FALLBACK_RELATIONS",
    "QUESTION_TYPE_PATTERNS",
    "QUESTION_TYPE_TO_RELATIONS",
    "RelationFilterResolution",
    "canonicalize_question_type",
    "filter_edges_by_question_type",
    "get_allowed_relations",
    "infer_question_type",
    "resolve_relation_filter",
    "summarize_relation_filter_stats",
]
