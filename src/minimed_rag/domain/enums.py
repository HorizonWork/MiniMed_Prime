"""Pseudocode enums for KG and reasoning semantics."""

from __future__ import annotations

from enum import StrEnum


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"


class Directionality(StrEnum):
    DIRECTED = "directed"
    UNDIRECTED = "undirected"


class PathType(StrEnum):
    POSITIVE = "positive"
    HARD_NEGATIVE = "hard_negative"
    RANDOM_NEGATIVE = "random_negative"
    PRUNED = "pruned"


class ValidityLabel(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
