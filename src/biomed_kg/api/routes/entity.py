"""Pseudocode API route: /entity/link."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from biomed_kg.api.dependencies import get_services

router = APIRouter()


@router.post("/entity/link")
def link_entity(services = Depends(get_services)):
    return {"linked_entities": []}
