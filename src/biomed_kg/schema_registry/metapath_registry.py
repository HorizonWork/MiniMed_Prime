"""Pseudocode metapath template registry."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from biomed_kg.common.config import load_yaml


@dataclass(frozen=True, slots=True)
class MetapathStep:
    subject_type: str
    predicate: str
    object_type: str


@dataclass(frozen=True, slots=True)
class MetapathTemplate:
    name: str
    task_type: str
    pattern: list[MetapathStep]
    max_depth: int
    priority: float
    min_edge_confidence: float


class MetapathRegistry:
    def __init__(self, path: str | Path = "configs/schema/metapath_templates.yaml"):
        raw = load_yaml(path).get("metapaths", {})
        self.templates = {}
        for name, values in raw.items():
            pattern = [MetapathStep(*step) for step in values["pattern"]]
            self.templates[name] = MetapathTemplate(name=name, pattern=pattern, **{k: v for k, v in values.items() if k != "pattern"})

    def get(self, template_id: str) -> MetapathTemplate:
        return self.templates[template_id]

    def get_for_task(self, task_type: str) -> list[MetapathTemplate]:
        return [template for template in self.templates.values() if template.task_type == task_type]
