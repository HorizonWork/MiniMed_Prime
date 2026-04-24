"""Pseudocode MedReason ingestion pipeline."""

from __future__ import annotations

from minimed_rag.ingestion.base import SourceIngestionPipeline
from minimed_rag.ingestion.medreason.normalizer import normalize_medreason_task
from minimed_rag.ingestion.medreason.parser import parse_medreason_tasks


class MedReasonIngestionPipeline(SourceIngestionPipeline):
    source_name = "medreason"

    def __init__(self, downloader=None, **kwargs):
        super().__init__(**kwargs)
        self.downloader = downloader

    def download(self):
        return self.downloader.download_release("local") if self.downloader else []

    def parse(self, raw_file):
        return parse_medreason_tasks(raw_file)

    def normalize(self, raw_record):
        return normalize_medreason_task(raw_record)
