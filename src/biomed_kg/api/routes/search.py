"""Pseudocode API route: /entity/search."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from biomed_kg.api.dependencies import get_services

router = APIRouter()


@router.post("/entity/search")
def search_entity(services = Depends(get_services)):
    return {"results": []}
