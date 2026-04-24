"""Pseudocode Prefect flow: generate_reasoning_data_flow.py."""
from __future__ import annotations

try:
    from prefect import flow
except Exception:
    def flow(fn=None, **kwargs):
        return fn if fn is not None else lambda wrapped: wrapped


@flow
def generate_reasoning_data_flow(graph_version: str = "kg_local"):
    steps = ['load_reasoning_tasks', 'find_paths', 'score_paths', 'sample_negatives', 'serialize_examples']
    executed_steps = []
    for step in steps:
        executed_steps.append({"step": step, "status": "planned"})
    return {"graph_version": graph_version, "steps": executed_steps}
