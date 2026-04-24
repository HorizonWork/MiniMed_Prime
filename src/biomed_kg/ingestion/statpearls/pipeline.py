"""Pseudocode StatPearls ingestion pipeline."""
from __future__ import annotations

from biomed_kg.ingestion.base import NormalizedRecord, RawRecord, SourceIngestionPipeline
from biomed_kg.ingestion.statpearls.normalizer import normalize_statpearls_chapter
from biomed_kg.ingestion.statpearls.parser import RawStatPearlsChapter, parse_statpearls_jsonl


class StatPearlsIngestionPipeline(SourceIngestionPipeline):
    source_name = "statpearls"

    def __init__(self, downloader=None, statpearls_chunker=None, **kwargs):
        super().__init__(**kwargs)
        self.downloader = downloader
        self.statpearls_chunker = statpearls_chunker

    def download(self):
        return self.downloader.download_release("local") if self.downloader else []

    def parse(self, raw_file):
        return [RawRecord("StatPearlsChapter", chapter.__dict__) for chapter in parse_statpearls_jsonl(raw_file)]

    def normalize(self, raw_record):
        chapter = RawStatPearlsChapter(**raw_record.payload)
        return [NormalizedRecord(type(item).__name__, item.__dict__ if hasattr(item, "__dict__") else item) for item in normalize_statpearls_chapter(chapter, self.statpearls_chunker)]
