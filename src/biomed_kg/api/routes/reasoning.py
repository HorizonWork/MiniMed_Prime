"""Pseudocode API route: /reasoning/find-paths."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from biomed_kg.api.dependencies import get_services

router = APIRouter()


@router.post("/reasoning/find-paths")
def find_paths(services = Depends(get_services)):
    return {"task_type": "unknown", "linked_entities": [], "paths": []}
