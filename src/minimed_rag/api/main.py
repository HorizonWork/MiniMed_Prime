"""Pseudocode FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI

from minimed_rag.api.routes import entity, graph, health, rag, reasoning, search, training_data

app = FastAPI(title="Biomedical KG-RAG")
for router_module in [health, entity, search, graph, rag, reasoning, training_data]:
    app.include_router(router_module.router)
