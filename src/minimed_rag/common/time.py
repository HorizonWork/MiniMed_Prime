"""Pseudocode for graph-version time helpers."""

from __future__ import annotations

from datetime import UTC, date, datetime


def utc_now() -> datetime:
    return datetime.now(UTC)


def current_graph_version(today: date | None = None) -> str:
    value = today or date.today()
    return f"kg_{value:%Y_%m_%d}"
