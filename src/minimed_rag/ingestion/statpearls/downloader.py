"""Pseudocode StatPearls downloader."""

from __future__ import annotations

from minimed_rag.ingestion.base import RawFile


class StatPearlsDownloader:
    def download_release(self, release: str) -> list[RawFile]:
        return [RawFile(name="statpearls_chapters.jsonl", release=release)]
