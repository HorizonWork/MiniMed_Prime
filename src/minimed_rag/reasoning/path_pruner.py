"""Path pruner — dedupe, score-floor, and top-k cap."""

from __future__ import annotations

from minimed_rag.domain.reasoning import ReasoningPath


class PathPruner:
    def __init__(
        self,
        min_score: float = 0.05,
        top_k: int | None = None,
    ) -> None:
        self.min_score = min_score
        self.top_k = top_k

    def prune(
        self,
        paths: list[ReasoningPath],
        plan=None,
        limit: int | None = None,
    ) -> list[ReasoningPath]:
        cap = limit if limit is not None else self.top_k
        filtered = [p for p in paths if p.path_confidence >= self.min_score]
        ordered = sorted(filtered, key=lambda p: p.path_confidence, reverse=True)
        if cap is not None and cap >= 0:
            ordered = ordered[:cap]
        return ordered

    @staticmethod
    def dedupe(paths: list[ReasoningPath]) -> list[ReasoningPath]:
        """Drop duplicate paths keyed by (template, predicate-tuple, endpoint-tuple).

        Keeps the first occurrence (preserves caller's order — typically
        descending by confidence after sorting).
        """
        seen: set[tuple] = set()
        result: list[ReasoningPath] = []
        for path in paths:
            key = (
                path.metapath_template_id,
                tuple(
                    (step.predicate, step.subject_entity_id, step.object_entity_id)
                    for step in path.steps
                ),
            )
            if key in seen:
                continue
            seen.add(key)
            result.append(path)
        return result

    @staticmethod
    def select_positive_paths(
        scored_paths: list[ReasoningPath],
        min_confidence: float = 0.5,
    ) -> list[ReasoningPath]:
        """Filter to high-confidence positive paths (Phase 7 dataset mode)."""
        return [p for p in scored_paths if p.path_confidence >= min_confidence]
