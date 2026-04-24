from __future__ import annotations

from abc import ABC, abstractmethod


class ModelProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> str: ...

    @property
    def config(self) -> dict:
        return {"provider": self.name}
