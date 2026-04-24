"""Pseudocode UMLS ingestion pipeline."""

from __future__ import annotations

from minimed_rag.ingestion.base import RawFile, RawRecord, SourceIngestionPipeline
from minimed_rag.ingestion.umls.mrconso_parser import parse_mrconso
from minimed_rag.ingestion.umls.mrdef_parser import parse_mrdef
from minimed_rag.ingestion.umls.mrhier_parser import parse_mrhier
from minimed_rag.ingestion.umls.mrrel_parser import parse_mrrel
from minimed_rag.ingestion.umls.mrsty_parser import parse_mrsty
from minimed_rag.ingestion.umls.normalizer import (
    normalize_umls_definition,
    normalize_umls_hierarchy,
    normalize_umls_mapping,
    normalize_umls_relation,
    normalize_umls_semantic_type,
    normalize_umls_term,
)


class UMLSIngestionPipeline(SourceIngestionPipeline):
    source_name = "umls"

    def download(self) -> list[RawFile]:
        return [
            RawFile(name)
            for name in [
                "MRCONSO.RRF",
                "MRSTY.RRF",
                "MRREL.RRF",
                "MRHIER.RRF",
                "MRDEF.RRF",
                "MRMAP.RRF",
            ]
        ]

    def parse(self, raw_file: RawFile) -> list[RawRecord]:
        parsers = {
            "MRCONSO.RRF": parse_mrconso,
            "MRSTY.RRF": parse_mrsty,
            "MRREL.RRF": parse_mrrel,
            "MRHIER.RRF": parse_mrhier,
            "MRDEF.RRF": parse_mrdef,
            "MRMAP.RRF": lambda file: [RawRecord("MRMAP", {"fields": [], "release": file.release})],
        }
        return parsers[raw_file.name](raw_file)

    def normalize(self, record: RawRecord):
        normalizers = {
            "MRCONSO": normalize_umls_term,
            "MRSTY": normalize_umls_semantic_type,
            "MRREL": normalize_umls_relation,
            "MRHIER": normalize_umls_hierarchy,
            "MRDEF": normalize_umls_definition,
            "MRMAP": normalize_umls_mapping,
        }
        return normalizers[record.type](record)
