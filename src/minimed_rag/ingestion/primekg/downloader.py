"""Pseudocode PrimeKG downloader."""

from __future__ import annotations

from minimed_rag.ingestion.base import RawFile


class PrimeKGDownloader:
    def download_release(self, release: str) -> list[RawFile]:
        return [RawFile(name="primekg_edges.csv", release=release)]
