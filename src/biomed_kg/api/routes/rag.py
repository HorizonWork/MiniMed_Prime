"""Pseudocode API route: /rag/answer."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from biomed_kg.api.dependencies import get_services

router = APIRouter()


@router.post("/rag/answer")
def rag_answer(services = Depends(get_services)):
    return {"answer": "", "citations": [], "confidence": 0.0}
