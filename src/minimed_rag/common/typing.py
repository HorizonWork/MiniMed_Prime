"""Shared pseudocode type aliases."""

from __future__ import annotations

from typing import Any, TypeAlias

JSONValue: TypeAlias = Any
Record: TypeAlias = dict[str, Any]
Batch: TypeAlias = list[Record]
