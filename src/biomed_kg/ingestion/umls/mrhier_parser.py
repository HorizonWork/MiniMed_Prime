"""Pseudocode parser for MRHIER."""
from __future__ import annotations

from biomed_kg.ingestion.base import RawFile, RawRecord
from biomed_kg.ingestion.umls.rrf_reader import read_rrf_lines


def parse_mrhier(raw_file: RawFile) -> list[RawRecord]:
    return [RawRecord(type="MRHIER", payload={"fields": fields, "release": raw_file.release}) for fields in read_rrf_lines(raw_file.bytes)]
