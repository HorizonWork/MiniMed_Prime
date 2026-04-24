"""Pseudocode chunk policy."""
from __future__ import annotations


class ChunkPolicy:
    def __init__(self, max_tokens: int = 384, overlap_tokens: int = 64):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def split_text(self, text: str) -> list[str]:
        return [text]
