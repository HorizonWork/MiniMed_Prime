from __future__ import annotations

import re
from collections.abc import Collection

NODE_TYPE_TO_ENTITY_TYPE = {
    "disease": "disease",
    "drug": "drug",
    "gene/protein": "protein",
    "protein": "protein",
    "effect": "symptom",
    "effect/phenotype": "symptom",
    "phenotype": "symptom",
    "symptom": "symptom",
    "anatomy": "anatomy",
}

LOOKUP_TOKEN_ALIASES = {
    "aery": "artery",
}

LOOKUP_PHRASE_ALIASES = {
    "colle s fascia": "colles fascia",
    "h pylori": "helicobacter pylori",
    "h pylori infectious disease": "helicobacter pylori infectious disease",
}

GENERIC_NAME_SUFFIXES = (
    " infectious disease",
    " disease",
    " disorder",
    " syndrome",
)


def normalize_basic_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalize_lookup_text(value: str) -> str:
    lowered = value.strip().lower()
    lowered = re.sub(r"\b([a-z0-9]+)'s\b", r"\1s", lowered)
    normalized = " ".join(re.findall(r"[a-z0-9]+", lowered))
    if not normalized:
        return ""
    normalized_tokens = [LOOKUP_TOKEN_ALIASES.get(token, token) for token in normalized.split()]
    normalized = " ".join(normalized_tokens)
    return LOOKUP_PHRASE_ALIASES.get(normalized, normalized)


def node_lookup_keys(value: str) -> set[str]:
    normalized = normalize_basic_text(value)
    lookup_normalized = normalize_lookup_text(value)
    keys = {key for key in (normalized, lookup_normalized) if key}
    if lookup_normalized:
        for suffix in GENERIC_NAME_SUFFIXES:
            if lookup_normalized.endswith(suffix):
                stripped = lookup_normalized[: -len(suffix)].strip()
                if stripped:
                    keys.add(stripped)
    return keys


def parse_node_cuis(value: str) -> list[str]:
    if not value:
        return []
    return [normalize_basic_text(candidate) for candidate in re.split(r"[|,; ]+", value) if normalize_basic_text(candidate)]


def resolve_primekg_node_identifier(
    *,
    node_index: str,
    node_id: str,
    duplicate_node_ids: Collection[str] | None = None,
) -> str:
    normalized_node_index = str(node_index).strip()
    normalized_node_id = str(node_id).strip() or f"node:{normalized_node_index}"
    if duplicate_node_ids is not None and normalized_node_id in duplicate_node_ids:
        return f"primekg_index:{normalized_node_index}"
    return normalized_node_id


__all__ = [
    "GENERIC_NAME_SUFFIXES",
    "LOOKUP_PHRASE_ALIASES",
    "LOOKUP_TOKEN_ALIASES",
    "NODE_TYPE_TO_ENTITY_TYPE",
    "node_lookup_keys",
    "normalize_basic_text",
    "normalize_lookup_text",
    "parse_node_cuis",
    "resolve_primekg_node_identifier",
]
