"""Pseudocode UMLS RRF reader."""

from __future__ import annotations


def read_rrf_lines(data: bytes) -> list[list[str]]:
    text = data.decode("utf-8", errors="replace")
    return [line.rstrip("|\n").split("|") for line in text.splitlines() if line]
