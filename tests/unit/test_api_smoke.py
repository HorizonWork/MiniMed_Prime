"""Smoke tests for FastAPI app."""

from __future__ import annotations

import asyncio

from httpx import ASGITransport, AsyncClient

from minimed_rag.api.main import app


async def _get_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get("/health")


def test_health_returns_200():
    response = asyncio.run(_get_health())
    assert response.status_code == 200


def test_health_returns_ok():
    response = asyncio.run(_get_health())
    body = response.json()
    assert body.get("status") == "ok"


def test_app_has_title():
    assert app.title == "Biomedical KG-RAG"
