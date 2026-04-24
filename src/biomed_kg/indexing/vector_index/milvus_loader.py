"""Pseudocode Milvus loader."""
from __future__ import annotations


class MilvusLoader:
    def __init__(self, milvus):
        self.milvus = milvus

    def load_records(self, collection: str, records: list[dict]) -> None:
        self.milvus.insert(collection=collection, records=records)
