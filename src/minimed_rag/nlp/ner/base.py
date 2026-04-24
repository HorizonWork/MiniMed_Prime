"""Pseudocode base NER interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Mention:
    text: str
    start: int
    end: int
    entity_type: str | None = None
    score: float = 0.0


class NERModel:
    def extract(self, text: str) -> list[Mention]:
        raise NotImplementedError
