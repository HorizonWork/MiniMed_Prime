"""Pseudocode task-aware query planner."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RetrievalPlan:
    query: str
    task_type: str
    linked_entities: list
    metapath_templates: list
    allowed_predicates: set[str] = field(default_factory=set)
    disallowed_predicates: set[str] = field(default_factory=set)
    max_depth: int = 1

    def get_entity_by_type(self, entity_type):
        wanted = set(entity_type if isinstance(entity_type, list) else [entity_type])
        for entity in self.linked_entities:
            if (
                getattr(entity, "entity_type", None) in wanted
                or getattr(entity, "semantic_type", None) in wanted
            ):
                return entity
        return self.linked_entities[0] if self.linked_entities else None


def collect_allowed_predicates(templates) -> set[str]:
    return {step.predicate for template in templates for step in template.pattern}


def collect_disallowed_predicates(task_type: str) -> set[str]:
    return set()


class QueryPlanner:
    def __init__(self, task_classifier, entity_linker, metapath_registry, ner):
        self.task_classifier = task_classifier
        self.entity_linker = entity_linker
        self.metapaths = metapath_registry
        self.ner = ner

    def plan(self, user_query: str) -> RetrievalPlan:
        task_type = self.task_classifier.classify(user_query)
        mentions = self.ner.extract(user_query)
        linked_entities = self.entity_linker.link_mentions(user_query, mentions)
        templates = self.metapaths.get_for_task(task_type)
        return RetrievalPlan(
            user_query,
            task_type,
            linked_entities,
            templates,
            collect_allowed_predicates(templates),
            collect_disallowed_predicates(task_type),
            max((template.max_depth for template in templates), default=1),
        )
