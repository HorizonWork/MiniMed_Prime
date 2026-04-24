"""Pseudocode object-store adapter for S3 or MinIO."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ObjectStore:
    bucket: str = "biomedkg"
    client: object | None = None
    memory: dict[str, bytes] = field(default_factory=dict)

    def put(self, path: str, data: bytes, metadata: dict | None = None) -> None:
        if self.client is None:
            self.memory[path] = data
            return
        self.client.put_object(Bucket=self.bucket, Key=path, Body=data, Metadata=metadata or {})

    def get(self, path: str) -> bytes:
        if self.client is None:
            return self.memory[path]
        response = self.client.get_object(Bucket=self.bucket, Key=path)
        return response["Body"].read()

    def list_prefix(self, prefix: str) -> list[str]:
        if self.client is None:
            return [key for key in self.memory if key.startswith(prefix)]
        return [obj["Key"] for obj in self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix).get("Contents", [])]
