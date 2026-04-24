"""Pseudocode path pruning."""

from __future__ import annotations


class PathPruner:
    def prune(self, paths: list, plan=None, limit: int = 50) -> list:
        filtered = [
            path
            for path in paths
            if getattr(path, "path_confidence", getattr(path, "score", 0.0)) > 0.0
        ]
        return sorted(
            filtered,
            key=lambda path: getattr(path, "path_confidence", getattr(path, "score", 0.0)),
            reverse=True,
        )[:limit]

    def select_positive_paths(self, scored_paths: list, min_confidence: float = 0.5) -> list:
        return [
            path for path in scored_paths if getattr(path, "path_confidence", 0.0) >= min_confidence
        ]
