"""Query planning.

Two coexisting layers:

- Phase 5 (this sprint): a lightweight ``QueryPlan`` enum
  (TEXT_ONLY / GRAPH_ONLY / HYBRID) plus ``decide_strategy(...)`` for the
  KG-augmented benchmark path. Heuristic-driven, zero metapath knowledge.
- Phase 6+: ``RetrievalPlan`` + ``QueryPlanner`` for full metapath-aware
  planning over the schema registry. Kept as scaffolding here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class QueryPlan(StrEnum):
    """Phase 5 retrieval strategy decision."""

    TEXT_ONLY = "text_only"
    GRAPH_ONLY = "graph_only"
    HYBRID = "hybrid"


QueryStrategy = QueryPlan


def decide_strategy(
    linked_entities,
    *,
    text_available: bool,
    graph_available: bool,
    min_entities_for_kg: int = 2,
) -> QueryPlan:
    """Pick the retrieval strategy for one question.

    Heuristic:

    - If neither side is wired, fall back to ``TEXT_ONLY`` (caller will
      degrade to no-retrieval at the next layer).
    - If text is unavailable but graph is + we have >=1 linked entity,
      use ``GRAPH_ONLY``.
    - If graph is unavailable, use ``TEXT_ONLY``.
    - If both are available and we have >=``min_entities_for_kg`` linked
      entities, prefer ``HYBRID`` — the per-spec "use both" path.
    - Otherwise use ``TEXT_ONLY``: below-threshold graph anchors are too weak
      for the Phase 5 hybrid path when text is available.
    """
    n = len(list(linked_entities)) if linked_entities is not None else 0

    if not text_available and not graph_available:
        return QueryPlan.TEXT_ONLY
    if not text_available and graph_available and n >= 1:
        return QueryPlan.GRAPH_ONLY
    if not graph_available:
        return QueryPlan.TEXT_ONLY
    if n >= min_entities_for_kg:
        return QueryPlan.HYBRID
    return QueryPlan.TEXT_ONLY


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
