"""Pseudocode API route: /training-data/generate."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from biomed_kg.api.dependencies import get_services

router = APIRouter()


@router.post("/training-data/generate")
def generate_training_data(services = Depends(get_services)):
    return {"status": "scheduled"}
