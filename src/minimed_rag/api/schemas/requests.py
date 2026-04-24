"""Pseudocode API request schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EntityLinkRequest(BaseModel):
    text: str


class EntitySearchRequest(BaseModel):
    query: str
    entity_types: list[str] | None = None


class GraphQueryRequest(BaseModel):
    cypher: str
    params: dict = {}


class GraphPathRequest(BaseModel):
    question: str
    task_type: str | None = None
    graph_version: str
    max_depth: int = Field(default=2, ge=1, le=5)


class RAGAnswerRequest(BaseModel):
    question: str
    graph_version: str | None = None


class TrainingDataGenerateRequest(BaseModel):
    dataset: str
    graph_version: str
    split: str = "train"
