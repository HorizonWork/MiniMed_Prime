"""Pseudocode Prefect flow: train_trm_flow.py."""
from __future__ import annotations

try:
    from prefect import flow
except Exception:
    def flow(fn=None, **kwargs):
        return fn if fn is not None else lambda wrapped: wrapped


@flow
def train_trm_flow(graph_version: str = "kg_local"):
    steps = ['load_dataset', 'train_model', 'evaluate_model', 'write_model_card']
    executed_steps = []
    for step in steps:
        executed_steps.append({"step": step, "status": "planned"})
    return {"graph_version": graph_version, "steps": executed_steps}
