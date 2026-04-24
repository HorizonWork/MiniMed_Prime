"""Smoke tests for FastAPI app."""

from __future__ import annotations

from fastapi.testclient import TestClient

from minimed_rag.api.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok():
    response = client.get("/health")
    body = response.json()
    assert body.get("status") == "ok"


def test_app_has_title():
    assert app.title == "Biomedical KG-RAG"
