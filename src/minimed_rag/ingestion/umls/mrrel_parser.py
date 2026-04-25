"""MRREL parser.

Filters out ``SUPPRESS != "N"`` and the relation classes we don't want to
materialize as graph edges:

- ``SY`` — synonymy; handled by ``Term``/``Concept`` nodes already
- ``AQ`` / ``QB`` — qualifier relations, rarely useful for QA
- ``SIB`` — sibling; redundant with the hierarchy relations
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from minimed_rag.ingestion.base import RawFile, RawRecord
from minimed_rag.ingestion.umls.mrrel_columns import MRREL_COLUMNS
from minimed_rag.ingestion.umls.rrf_reader import iter_rrf_lines, read_rrf_lines

SKIP_RELS = frozenset({"SY", "AQ", "QB", "SIB"})


def iter_mrrel(
    path: str | Path,
    limit: int | None = None,
) -> Iterator[dict[str, str]]:
    for row in iter_rrf_lines(path, MRREL_COLUMNS, limit=limit):
        if row["SUPPRESS"] != "N":
            continue
        if row["REL"] in SKIP_RELS:
            continue
        # Skip self-loops
        if row["CUI1"] == row["CUI2"]:
            continue
        yield {
            "cui1": row["CUI1"],
            "cui2": row["CUI2"],
            "rel": row["REL"],
            "rela": row["RELA"],
            "rui": row["RUI"],
            "sab": row["SAB"],
            "dir": row["DIR"],
        }


def parse_mrrel(raw_file: RawFile) -> list[RawRecord]:
    records: list[RawRecord] = []
    for fields in read_rrf_lines(raw_file.bytes):
        if len(fields) != len(MRREL_COLUMNS):
            continue
        row = dict(zip(MRREL_COLUMNS, fields, strict=True))
        if row["SUPPRESS"] != "N" or row["REL"] in SKIP_RELS:
            continue
        if row["CUI1"] == row["CUI2"]:
            continue
        records.append(
            RawRecord(
                type="MRREL",
                payload={
                    "cui1": row["CUI1"],
                    "cui2": row["CUI2"],
                    "rel": row["REL"],
                    "rela": row["RELA"],
                    "rui": row["RUI"],
                    "sab": row["SAB"],
                    "release": raw_file.release,
                },
            )
        )
    return records
