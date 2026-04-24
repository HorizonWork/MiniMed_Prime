"""Pseudocode Milvus vector adapter."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Milvus:
    client: object

    def ensure_collection(self, name: str, schema: dict) -> None:
        if not self.client.has_collection(name):
            self.client.create_collection(collection_name=name, schema=schema)

    def insert(self, collection: str, records: list[dict]) -> None:
        if records:
            self.client.insert(collection_name=collection, data=records)

    def search(self, collection: str, vector: list[float], top_k: int, filters: dict | None = None):
        return self.client.search(collection_name=collection, data=[vector], limit=top_k, filter=filters)
