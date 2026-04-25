"""MRSTY parser — UMLS semantic-type assignments."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from minimed_rag.ingestion.base import RawFile, RawRecord
from minimed_rag.ingestion.umls.mrsty_columns import MRSTY_COLUMNS
from minimed_rag.ingestion.umls.rrf_reader import iter_rrf_lines, read_rrf_lines


def iter_mrsty(
    path: str | Path,
    limit: int | None = None,
) -> Iterator[dict[str, str]]:
    for row in iter_rrf_lines(path, MRSTY_COLUMNS, limit=limit):
        yield {
            "cui": row["CUI"],
            "tui": row["TUI"],
            "sty": row["STY"],
        }


def parse_mrsty(raw_file: RawFile) -> list[RawRecord]:
    records: list[RawRecord] = []
    for fields in read_rrf_lines(raw_file.bytes):
        if len(fields) != len(MRSTY_COLUMNS):
            continue
        row = dict(zip(MRSTY_COLUMNS, fields, strict=True))
        records.append(
            RawRecord(
                type="MRSTY",
                payload={
                    "cui": row["CUI"],
                    "tui": row["TUI"],
                    "sty": row["STY"],
                    "release": raw_file.release,
                },
            )
        )
    return records
