"""Pseudocode PrimeKG CSV parser."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO

from minimed_rag.ingestion.base import RawFile


@dataclass(slots=True)
class RawPrimeKGEdge:
    source_node_id: str
    source_node_name: str
    source_node_type: str
    relation: str
    target_node_id: str
    target_node_name: str
    target_node_type: str
    source: str | None = None


def parse_primekg_edges(raw_file: RawFile) -> list[RawPrimeKGEdge]:
    rows = csv.DictReader(StringIO(raw_file.bytes.decode("utf-8")))
    records = []
    for row in rows:
        records.append(
            RawPrimeKGEdge(
                row["x_id"],
                row["x_name"],
                row["x_type"],
                row["relation"],
                row["y_id"],
                row["y_name"],
                row["y_type"],
                row.get("source"),
            )
        )
    return records
