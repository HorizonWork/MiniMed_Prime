"""Pseudocode merge policy."""
from __future__ import annotations


class MergePolicy:
    def allow_merge(self, left, right) -> bool:
        if getattr(left, "cui", None) and left.cui == getattr(right, "canonical_cui", None):
            return True
        return getattr(left, "entity_type", None) == getattr(right, "entity_type", None) and left.name.casefold() == right.preferred_name.casefold()
