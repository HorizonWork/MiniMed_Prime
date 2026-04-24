"""Pseudocode for source and text hashing."""
from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str) -> str:
    return hash_bytes(text.encode("utf-8"))


def hash_record(record: Any) -> str:
    payload = record.__dict__ if hasattr(record, "__dict__") else record
    return hash_text(json.dumps(payload, sort_keys=True, default=str))
