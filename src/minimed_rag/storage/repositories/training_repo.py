"""Pseudocode repository for serialized TRM examples."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TrainingRepo:
    examples: list[dict] = field(default_factory=list)

    def write_examples(self, examples: list[dict]) -> None:
        self.examples.extend(examples)

    def load_examples(self, split: str, graph_version: str) -> list[dict]:
        return [
            ex
            for ex in self.examples
            if ex.get("split", split) == split and ex.get("graph_version") == graph_version
        ]
