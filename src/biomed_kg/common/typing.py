"""Shared pseudocode type aliases."""
from __future__ import annotations

from typing import Any, TypeAlias

JSONValue: TypeAlias = type(None) | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
Record: TypeAlias = dict[str, Any]
Batch: TypeAlias = list[Record]
