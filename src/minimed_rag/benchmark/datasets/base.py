from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MCQExample:
    id: str
    question: str
    options: dict[str, str]  # {"A": "...", "B": "...", ...}
    answer: str  # "A", "B", "yes", "no", "maybe"
    dataset: str
    context: str | None = None


class BaseDataset(ABC):
    name: str

    @abstractmethod
    def load(self, split: str = "test") -> list[MCQExample]: ...
