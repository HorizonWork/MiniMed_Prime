"""Pseudocode MedReason downloader."""
from __future__ import annotations

from biomed_kg.ingestion.base import RawFile


class MedReasonDownloader:
    def download_release(self, release: str) -> list[RawFile]:
        return [RawFile(name="medreason_tasks.jsonl", release=release)]
