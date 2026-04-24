"""Pseudocode Milvus collection schema."""

from __future__ import annotations


def chunk_collection_schema(dim: int) -> dict:
    return {
        "collection_name": "minimed_chunk_dense_v1",
        "fields": [
            {"name": "vector_id", "type": "varchar", "primary": True},
            {"name": "vector", "type": "float_vector", "dim": dim},
        ],
    }
