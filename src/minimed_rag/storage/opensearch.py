"""Pseudocode OpenSearch lexical adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class OpenSearch:
    client: object

    def ensure_index(self, index: str, body: dict) -> None:
        if not self.client.indices.exists(index=index):
            self.client.indices.create(index=index, body=body)

    def bulk_index(self, index: str, docs: list[dict]) -> None:
        body = []
        for doc in docs:
            doc_id = doc.pop("_id", None)
            body.append({"index": {"_index": index, "_id": doc_id}})
            body.append(doc)
        if body:
            self.client.bulk(body=body, refresh=False)

    def search(self, index: str, body: dict):
        return self.client.search(index=index, body=body)
