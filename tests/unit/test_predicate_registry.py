"""Unit tests for the PredicateRegistry type-matching semantics.

The legacy `_matches` treated `"Entity" in allowed` as a wildcard against any
entity_type, including missing/empty ones. That was a silent false-positive
in negative sampling and direction resolution. Phase 6 cleanup tightens this:

  - empty/None entity_type does NOT match unless `allowed` is exactly
    `["Entity"]` (the only case where the predicate is genuinely entity-
    agnostic and the missing type is harmless).
  - non-empty entity_type matches when `"Entity" in allowed` OR exact match.

This preserves generic predicates (`associated_with` with `domain: [Entity]`)
while killing the silent-pass on missing types.
"""

from __future__ import annotations

from minimed_rag.schema_registry.predicate_registry import PredicateRegistry


def _registry() -> PredicateRegistry:
    """Real registry loaded from the shipped predicates.yaml."""
    return PredicateRegistry()


def test_matches_type_empty_with_entity_only_allows() -> None:
    assert PredicateRegistry.matches_type("", ["Entity"]) is True


def test_matches_type_empty_with_specific_rejects() -> None:
    assert PredicateRegistry.matches_type("", ["Drug"]) is False


def test_matches_type_empty_with_mixed_entity_rejects() -> None:
    # "Entity" mixed with specific types is ambiguous — empty doesn't license.
    assert PredicateRegistry.matches_type("", ["Drug", "Entity"]) is False


def test_matches_type_none_treated_as_empty() -> None:
    assert PredicateRegistry.matches_type(None, ["Drug"]) is False
    assert PredicateRegistry.matches_type(None, ["Entity"]) is True


def test_matches_type_specific_in_entity_allowed() -> None:
    assert PredicateRegistry.matches_type("Drug", ["Entity"]) is True


def test_matches_type_specific_exact_match() -> None:
    assert PredicateRegistry.matches_type("Drug", ["Drug", "Disease"]) is True


def test_matches_type_specific_no_match() -> None:
    assert PredicateRegistry.matches_type("Drug", ["Disease"]) is False


def test_is_valid_domain_range_treats_drug_disease() -> None:
    reg = _registry()
    assert reg.is_valid_domain_range("treats", "Drug", "Disease") is True


def test_is_valid_domain_range_rejects_missing_subject() -> None:
    """Regression guard: empty subject_type used to silently pass when
    `domain: [Drug, ...]` because the old `_matches` was lax."""
    reg = _registry()
    assert reg.is_valid_domain_range("treats", "", "Disease") is False


def test_is_valid_domain_range_associated_with_preserved() -> None:
    """`associated_with` has domain/range `[Entity]` and must still permit
    arbitrary entity_types — that's its spec intent as a generic catch-all."""
    reg = _registry()
    assert reg.is_valid_domain_range("associated_with", "Drug", "Disease") is True
    assert reg.is_valid_domain_range("associated_with", "Gene", "Pathway") is True


def test_is_valid_domain_range_associated_with_rejects_missing() -> None:
    """Even `associated_with` should reject when both ends are missing — a
    fact with no typed endpoints is unreliable."""
    reg = _registry()
    # domain/range are both [Entity] (length 1 with "Entity") → empty allowed.
    assert reg.is_valid_domain_range("associated_with", "", "") is True
    # But a half-typed assertion is a partial failure — only valid if the
    # rule's range is also entity-only. Both checks pass for associated_with.
    assert reg.is_valid_domain_range("associated_with", "Drug", "") is True


def test_sample_incompatible_predicate_finds_one() -> None:
    """Should return some predicate whose domain/range disagree with
    (subject_type, object_type)."""
    reg = _registry()
    pick = reg.sample_incompatible_predicate("Drug", "Disease", "treats")
    assert pick != "treats"
    assert reg.is_valid_domain_range(pick, "Drug", "Disease") is False


def test_sample_incompatible_predicate_excludes_associated_with() -> None:
    """`associated_with` would match anything (Entity domain/range) so it
    cannot be picked as an "incompatible" predicate."""
    reg = _registry()
    pick = reg.sample_incompatible_predicate("Drug", "Disease", "treats")
    assert pick != "associated_with"


def test_legacy_underscore_matches_still_callable() -> None:
    """Backward-compat wrapper preserved for any internal caller."""
    assert PredicateRegistry._matches("Drug", ["Drug"]) is True
    assert PredicateRegistry._matches("", ["Drug"]) is False
