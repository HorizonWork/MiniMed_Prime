"""Reasoning path scorer.

Combines per-edge confidence, metapath template prior, entity-type
agreement, length decay, direction penalty (for genuinely reversed
steps, not template-required inversions), an
``associated_with``-overuse penalty, and a hub penalty (paths through
high-degree intermediate entities are demoted) into a single
``path_confidence`` in ``[0, 1]``.

Hub penalty is gated on an :class:`EntityStatsRepo` snapshot. When no
repo is supplied, ``hub_penalty`` is a no-op (1.0) and the scorer
behaves identically to its pre-Phase-6-cleanup form. ``evidence_coverage``
is still omitted — PrimeKG has no per-assertion ``evidence_ids``;
re-introduce when Phase 5+ ingests PubMed/StatPearls.
"""

from __future__ import annotations

import math
from typing import Any

from minimed_rag.domain.reasoning import ReasoningPath
from minimed_rag.schema_registry.metapath_registry import MetapathRegistry, MetapathTemplate


class PathScorer:
    def __init__(
        self,
        metapath_registry: MetapathRegistry,
        entity_stats_repo: Any | None = None,
        hub_snapshot_id: str = "",
    ) -> None:
        self.metapath_registry = metapath_registry
        self.entity_stats_repo = entity_stats_repo
        self.hub_snapshot_id = hub_snapshot_id
        self._p50_cached: float | None = None

    def score(self, paths: list[ReasoningPath], task=None) -> list[ReasoningPath]:
        for path in paths:
            template = self._safe_template(path.metapath_template_id)
            metapath_prior = template.priority if template is not None else 0.5

            edge_confs = [step.edge_confidence for step in path.steps]
            min_conf = min(edge_confs) if edge_confs else 0.0

            type_match = self._calculate_type_match(path, template)
            length_decay = 1.0 / (1.0 + 0.2 * max(0, len(path.steps) - 1))
            direction_score = self._calculate_direction_score(path)
            assoc_penalty = self._calculate_associated_with_penalty(path)
            hub_penalty = self._calculate_hub_penalty(path)

            base = 0.5 * min_conf + 0.3 * type_match + 0.2 * length_decay
            score = metapath_prior * direction_score * assoc_penalty * hub_penalty * base
            path.path_confidence = max(0.0, min(1.0, score))
        return paths

    def _safe_template(self, template_id: str) -> MetapathTemplate | None:
        if not template_id:
            return None
        try:
            return self.metapath_registry.get(template_id)
        except (KeyError, AttributeError, TypeError):
            return None

    @staticmethod
    def _calculate_direction_score(path: ReasoningPath) -> float:
        # ``template_inverse`` is a legitimate traversal of the canonical
        # predicate against its stored direction (e.g. ``target_of`` of a
        # ``targets`` assertion); no penalty.
        if any(step.direction == "reversed" for step in path.steps):
            return 0.6
        return 1.0

    @staticmethod
    def _calculate_associated_with_penalty(path: ReasoningPath) -> float:
        count = sum(1 for step in path.steps if step.predicate == "associated_with")
        if count == 0:
            return 1.0
        return max(0.4, 1.0 - 0.25 * count)

    @staticmethod
    def _calculate_type_match(path: ReasoningPath, template: MetapathTemplate | None) -> float:
        if template is None or not path.steps:
            # Template-free expansion: no template constraint to violate.
            return 1.0
        pattern = getattr(template, "pattern", None) or ()
        if pattern and len(pattern) != len(path.steps):
            return 0.5
        # We only know the entity_id of each endpoint, not its type, from the
        # ReasoningPathStep. The template-aware Cypher already filtered to
        # matching types, so reaching this code at all means the types match;
        # we award full credit. Future: stamp entity_type on the step itself
        # so we can verify here.
        return 1.0

    def _calculate_hub_penalty(self, path: ReasoningPath) -> float:
        """Per-intermediate-hop hub penalty.

        Continuous, scale-invariant: ``1 / (1 + log10(1 + degree / p50))``
        per intermediate entity, multiplied across the path. Endpoints
        (anchor + terminal) are NOT penalized — being a well-known
        question entity or candidate answer is not a hub problem.

        Returns 1.0 if no stats repo, no hub_snapshot_id, p50 is degenerate
        (≤ 0), or the path has fewer than 3 entities (no intermediates).
        Missing degree lookups are skipped, not punished as zero.
        """
        if self.entity_stats_repo is None or not self.hub_snapshot_id:
            return 1.0
        if not path.steps:
            return 1.0
        intermediates = [step.object_entity_id for step in path.steps[:-1]]
        if not intermediates:
            return 1.0
        p50 = self._p50_degree()
        if p50 <= 0.0:
            return 1.0
        factor = 1.0
        for entity_id in intermediates:
            if not entity_id:
                continue
            degree = self.entity_stats_repo.get_degree(entity_id, self.hub_snapshot_id)
            if degree is None:
                continue
            ratio = degree / p50
            factor *= 1.0 / (1.0 + math.log10(1.0 + ratio))
        return factor

    def _p50_degree(self) -> float:
        if self._p50_cached is None:
            self._p50_cached = self.entity_stats_repo.get_p50_degree(self.hub_snapshot_id)
        return self._p50_cached
