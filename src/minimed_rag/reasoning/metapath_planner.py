"""Pseudocode metapath planner."""

from __future__ import annotations


class MetapathPlanner:
    def __init__(self, metapath_registry):
        self.metapath_registry = metapath_registry

    def plan_for_task(self, task_type: str) -> list:
        return sorted(
            self.metapath_registry.get_for_task(task_type),
            key=lambda template: template.priority,
            reverse=True,
        )
