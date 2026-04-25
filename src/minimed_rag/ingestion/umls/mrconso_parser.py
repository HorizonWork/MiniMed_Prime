"""MRCONSO parser.

Filters to ENG + SUPPRESS=N and yields structured records suitable for
downstream concept/term/crosswalk loading.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from minimed_rag.ingestion.base import RawFile, RawRecord
from minimed_rag.ingestion.umls.mrconso_columns import MRCONSO_COLUMNS
from minimed_rag.ingestion.umls.rrf_reader import iter_rrf_lines, read_rrf_lines


def iter_mrconso(
    path: str | Path,
    languages: tuple[str, ...] = ("ENG",),
    limit: int | None = None,
) -> Iterator[dict[str, str]]:
    """Stream MRCONSO rows as structured dicts.

    Output keys: ``cui, lat, ts, aui, sab, tty, code, str, is_preferred,
    suppress``. Rows with ``SUPPRESS != "N"`` or ``LAT`` outside ``languages``
    are filtered out.
    """
    for row in iter_rrf_lines(path, MRCONSO_COLUMNS, limit=limit):
        if row["SUPPRESS"] != "N":
            continue
        if row["LAT"] not in languages:
            continue
        yield {
            "cui": row["CUI"],
            "lat": row["LAT"],
            "ts": row["TS"],
            "aui": row["AUI"],
            "sab": row["SAB"],
            "tty": row["TTY"],
            "code": row["CODE"],
            "str": row["STR"],
            "is_preferred": row["ISPREF"] == "Y" and row["TS"] == "P",
            "suppress": row["SUPPRESS"],
        }


def parse_mrconso(raw_file: RawFile) -> list[RawRecord]:
    """Legacy bytes-in interface — small-input / test usage only."""
    records: list[RawRecord] = []
    for fields in read_rrf_lines(raw_file.bytes):
        if len(fields) != len(MRCONSO_COLUMNS):
            continue
        row = dict(zip(MRCONSO_COLUMNS, fields, strict=True))
        if row["SUPPRESS"] != "N" or row["LAT"] != "ENG":
            continue
        records.append(
            RawRecord(
                type="MRCONSO",
                payload={
                    "cui": row["CUI"],
                    "aui": row["AUI"],
                    "sab": row["SAB"],
                    "tty": row["TTY"],
                    "code": row["CODE"],
                    "str": row["STR"],
                    "lat": row["LAT"],
                    "is_preferred": row["ISPREF"] == "Y" and row["TS"] == "P",
                    "release": raw_file.release,
                },
            )
        )
    return records
