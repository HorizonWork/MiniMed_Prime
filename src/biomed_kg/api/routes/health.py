"""Pseudocode API route: /health."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from biomed_kg.api.dependencies import get_services

router = APIRouter()


@router.get("/health")
def health(services = Depends(get_services)):
    return {"status": "ok"}
