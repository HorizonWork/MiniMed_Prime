"""Pseudocode PrimeKG ingestion pipeline."""

from __future__ import annotations

from minimed_rag.ingestion.base import RawRecord, SourceIngestionPipeline
from minimed_rag.ingestion.primekg.normalizer import normalize_primekg_edge
from minimed_rag.ingestion.primekg.parser import RawPrimeKGEdge, parse_primekg_edges


class PrimeKGIngestionPipeline(SourceIngestionPipeline):
    source_name = "primekg"

    def __init__(self, predicate_mapper, downloader=None, **kwargs):
        super().__init__(**kwargs)
        self.predicate_mapper = predicate_mapper
        self.downloader = downloader

    def download(self):
        return self.downloader.download_release("local") if self.downloader else []

    def parse(self, raw_file):
        return [RawRecord("PrimeKGEdge", edge.__dict__) for edge in parse_primekg_edges(raw_file)]

    def normalize(self, record):
        return normalize_primekg_edge(RawPrimeKGEdge(**record.payload), self.predicate_mapper)
