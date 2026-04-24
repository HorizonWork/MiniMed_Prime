"""Pseudocode PubMed baseline loader."""

from __future__ import annotations


class PubMedBaselineLoader:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self, release: str) -> None:
        self.pipeline.run_baseline(release=release)
