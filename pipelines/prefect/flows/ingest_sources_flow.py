"""Pseudocode Prefect flow: ingest_sources_flow.py."""
from __future__ import annotations

try:
    from prefect import flow
except Exception:
    def flow(fn=None, **kwargs):
        return fn if fn is not None else lambda wrapped: wrapped


@flow
def ingest_sources_flow(graph_version: str = "kg_local"):
    steps = ['ingest_umls', 'ingest_primekg', 'ingest_pubmed', 'ingest_statpearls', 'ingest_medreason']
    executed_steps = []
    for step in steps:
        executed_steps.append({"step": step, "status": "planned"})
    return {"graph_version": graph_version, "steps": executed_steps}
