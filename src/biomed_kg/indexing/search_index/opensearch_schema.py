"""Pseudocode OpenSearch index schema."""
from __future__ import annotations


def chunk_index_schema() -> dict:
    return {"mappings": {"properties": {"text": {"type": "text"}, "title": {"type": "text"}, "entity_ids": {"type": "keyword"}, "graph_version": {"type": "keyword"}}}}
