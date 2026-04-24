"""Pseudocode reasoning path quality evaluator."""
from __future__ import annotations


class PathQualityEvaluator:
    def evaluate(self, paths: list) -> dict:
        total = len(paths)
        valid = sum(1 for path in paths if getattr(path, "validity_label", None) == "valid")
        associated = sum(1 for path in paths for step in getattr(path, "steps", []) if step.predicate == "associated_with")
        return {"path_found_rate": float(total > 0), "valid_path_rate": valid / total if total else 0.0, "associated_with_overuse_rate": associated / total if total else 0.0}
