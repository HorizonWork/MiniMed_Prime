"""Pseudocode API response schemas."""
from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"


class LinkedEntityResponse(BaseModel):
    text: str
    entity_id: str
    confidence: float


class PathStepResponse(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float


class ReasoningPathResponse(BaseModel):
    path_id: str
    path_confidence: float
    steps: list[PathStepResponse]
    evidence_ids: list[str] = []


class RAGAnswerResponse(BaseModel):
    answer: str
    citations: list[str]
    confidence: float
