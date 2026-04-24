"""Pseudocode SciSpacy NER wrapper."""
from __future__ import annotations

from biomed_kg.nlp.ner.base import Mention


class SciSpacyNER:
    def __init__(self, nlp):
        self.nlp = nlp

    def extract(self, text: str) -> list[Mention]:
        doc = self.nlp(text)
        return [Mention(ent.text, ent.start_char, ent.end_char, ent.label_, 1.0) for ent in doc.ents]
