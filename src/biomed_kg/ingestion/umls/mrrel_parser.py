"""Pseudocode parser for MRREL."""
from __future__ import annotations

from biomed_kg.ingestion.base import RawFile, RawRecord
from biomed_kg.ingestion.umls.rrf_reader import read_rrf_lines


def parse_mrrel(raw_file: RawFile) -> list[RawRecord]:
    return [RawRecord(type="MRREL", payload={"fields": fields, "release": raw_file.release}) for fields in read_rrf_lines(raw_file.bytes)]
