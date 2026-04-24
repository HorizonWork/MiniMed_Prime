"""Pseudocode confidence policy loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from minimed_rag.common.config import load_yaml


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    weights: dict[str, float]
    source_reliability: dict[str, float]
    path_confidence: dict

    @classmethod
    def load(cls, path: str | Path = "configs/schema/confidence_policy.yaml") -> ConfidencePolicy:
        raw = load_yaml(path)
        return cls(
            weights=raw.get("assertion_confidence", {}).get("weights", {}),
            source_reliability=raw.get("source_reliability", {}),
            path_confidence=raw.get("path_confidence", {}),
        )
