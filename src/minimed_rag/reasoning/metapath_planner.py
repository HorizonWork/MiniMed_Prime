"""Metapath planner.

Selects ordered ``MetapathTemplate``s for a given task type from the
``MetapathRegistry``. Sorting is by descending priority with a stable
tie-break on template name so a single ordering is observed across runs.
"""

from __future__ import annotations

from minimed_rag.schema_registry.metapath_registry import MetapathRegistry, MetapathTemplate


class MetapathPlanner:
    def __init__(self, metapath_registry: MetapathRegistry) -> None:
        self.metapath_registry = metapath_registry

    def plan_for_task(self, task_type: str) -> list[MetapathTemplate]:
        templates = self.metapath_registry.get_for_task(task_type)
        return sorted(templates, key=lambda t: (-t.priority, t.name))
