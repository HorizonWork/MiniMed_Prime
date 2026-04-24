"""Pseudocode parser for MRCONSO."""
from __future__ import annotations

from biomed_kg.ingestion.base import RawFile, RawRecord
from biomed_kg.ingestion.umls.rrf_reader import read_rrf_lines


def parse_mrconso(raw_file: RawFile) -> list[RawRecord]:
    return [RawRecord(type="MRCONSO", payload={"fields": fields, "release": raw_file.release}) for fields in read_rrf_lines(raw_file.bytes)]
