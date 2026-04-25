"""Reasoning path generation (Phase 6).

Single-question pipeline:
    TaskClassifier -> MetapathPlanner -> PathFinder -> PathScorer -> PathPruner
        -> (optional) NegativeSampler

The CLI entry point is ``minimed generate-paths``.
"""
