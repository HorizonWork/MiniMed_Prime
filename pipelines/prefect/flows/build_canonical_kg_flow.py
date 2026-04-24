"""Pseudocode Prefect flow: build_canonical_kg_flow.py."""
from __future__ import annotations

try:
    from prefect import flow
except Exception:
    def flow(fn=None, **kwargs):
        return fn if fn is not None else lambda wrapped: wrapped


@flow
def build_canonical_kg_flow(graph_version: str = "kg_local"):
    steps = ['build_terminology_layer', 'normalize_sources', 'resolve_entities', 'build_assertions', 'validate_canonical_kg']
    executed_steps = []
    for step in steps:
        executed_steps.append({"step": step, "status": "planned"})
    return {"graph_version": graph_version, "steps": executed_steps}
