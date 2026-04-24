"""Pseudocode PubMed daily update loader."""

from __future__ import annotations


class PubMedDailyUpdateLoader:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self, update_file) -> None:
        self.pipeline.run_daily_update(update_file)
