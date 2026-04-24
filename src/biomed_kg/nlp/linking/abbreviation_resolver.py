"""Pseudocode abbreviation resolver."""
from __future__ import annotations

import re


class AbbreviationResolver:
    def resolve(self, text: str) -> dict[str, str]:
        pairs = {}
        for match in re.finditer(r"([A-Za-z][A-Za-z -]+) \(([A-Z0-9-]{2,})\)", text):
            pairs[match.group(2)] = match.group(1).strip()
        return pairs
