"""Pseudocode API route: /graph/paths."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from minimed_rag.api.dependencies import get_services

router = APIRouter()


@router.post("/graph/paths")
def graph_paths(services=Depends(get_services)):
    return {"paths": []}
