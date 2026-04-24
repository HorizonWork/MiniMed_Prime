"""Pseudocode base ingestion interface."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from minimed_rag.common.hashing import hash_bytes


@dataclass(slots=True)
class RawFile:
    name: str
    bytes: bytes = b""
    release: str = "local"
    checksum: str | None = None

    def __post_init__(self) -> None:
        if self.checksum is None:
            self.checksum = hash_bytes(self.bytes)


@dataclass(slots=True)
class RawRecord:
    type: str
    payload: dict[str, Any]


@dataclass(slots=True)
class NormalizedRecord:
    record_type: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class SourceIngestionPipeline:
    source_name: str = "source"

    def __init__(self, object_store=None, lakehouse=None):
        self.object_store = object_store
        self.lakehouse = lakehouse

    def download(self) -> list[RawFile]:
        raise NotImplementedError

    def parse(self, raw_file: RawFile) -> list[RawRecord]:
        raise NotImplementedError

    def normalize(self, raw_record: RawRecord) -> list[NormalizedRecord]:
        raise NotImplementedError

    def write_raw(self, raw_file: RawFile) -> None:
        if not self.object_store:
            return
        self.object_store.put(
            path=f"bronze/{self.source_name}/{raw_file.release}/{raw_file.name}",
            data=raw_file.bytes,
            metadata={
                "checksum": raw_file.checksum,
                "source": self.source_name,
                "release": raw_file.release,
            },
        )

    def write_normalized(self, records: Iterable[NormalizedRecord]) -> None:
        if self.lakehouse:
            self.lakehouse.write(
                table=f"silver_{self.source_name}_records", records=records, mode="append"
            )

    def run(self) -> None:
        for raw_file in self.download():
            self.write_raw(raw_file)
            normalized_records = []
            for raw_record in self.parse(raw_file):
                normalized_records.extend(self.normalize(raw_record))
            self.write_normalized(normalized_records)
