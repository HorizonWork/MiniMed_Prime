"""Pseudocode Prefect flow: build_serving_indexes_flow.py."""
from __future__ import annotations

try:
    from prefect import flow
except Exception:
    def flow(fn=None, **kwargs):
        return fn if fn is not None else lambda wrapped: wrapped


@flow
def build_serving_indexes_flow(graph_version: str = "kg_local"):
    steps = ['build_neo4j_projection', 'build_vector_indexes', 'build_opensearch_indexes']
    executed_steps = []
    for step in steps:
        executed_steps.append({"step": step, "status": "planned"})
    return {"graph_version": graph_version, "steps": executed_steps}
