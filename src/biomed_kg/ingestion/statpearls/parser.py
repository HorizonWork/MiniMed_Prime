"""Pseudocode StatPearls parser."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from biomed_kg.ingestion.base import RawFile


@dataclass(slots=True)
class RawStatPearlsChapter:
    nbk_id: str
    title: str
    sections: list
    authors: list[str] = field(default_factory=list)
    updated_at: object | None = None
    license: str | None = None


def parse_statpearls_jsonl(raw_file: RawFile) -> list[RawStatPearlsChapter]:
    chapters = []
    for line in raw_file.bytes.decode("utf-8").splitlines():
        payload = json.loads(line)
        chapters.append(RawStatPearlsChapter(**payload))
    return chapters
