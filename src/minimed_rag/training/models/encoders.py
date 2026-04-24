"""Pseudocode text and vocab encoders."""

from __future__ import annotations


class TokenEncoder:
    def encode(self, text: str) -> list[int]:
        return [hash(token) % 30000 for token in text.split()]


class Vocab:
    def __init__(self):
        self.ids = {}

    @property
    def size(self) -> int:
        return len(self.ids)

    def get_id(self, value: str) -> int:
        if value not in self.ids:
            self.ids[value] = len(self.ids)
        return self.ids[value]
