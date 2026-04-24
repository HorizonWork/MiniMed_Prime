from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


class CorpusLoader(ABC):
    name: str

    @abstractmethod
    def load(self, limit: int = 0) -> list[Chunk]: ...
