"""Pseudocode MedReason parser."""

from __future__ import annotations

import json

from minimed_rag.ingestion.base import RawFile, RawRecord


def parse_medreason_tasks(raw_file: RawFile) -> list[RawRecord]:
    return [
        RawRecord("MedReasonTask", json.loads(line))
        for line in raw_file.bytes.decode("utf-8").splitlines()
        if line
    ]
