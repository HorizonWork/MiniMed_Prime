"""Pseudocode PubMed downloader."""

from __future__ import annotations

from datetime import date

from minimed_rag.ingestion.base import RawFile


class PubMedDownloader:
    def download_baseline_files(self, release: str) -> list[RawFile]:
        return [RawFile(name="pubmed_baseline.xml.gz", release=release)]

    def download_daily_update(self, update_date: date) -> RawFile:
        return RawFile(
            name=f"pubmed_update_{update_date:%Y_%m_%d}.xml.gz", release=f"{update_date:%Y_%m_%d}"
        )
