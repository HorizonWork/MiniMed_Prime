from __future__ import annotations

from .base import ModelProvider


class OllamaProvider(ModelProvider):
    """Calls a locally-running Ollama server (no paid API required)."""

    def __init__(
        self,
        model: str = "phi3:mini",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self.name = f"ollama:{model}"

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        import json
        import urllib.request

        payload = json.dumps(
            {
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": self._temperature,
                },
            }
        ).encode()

        req = urllib.request.Request(
            f"{self._base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        return body.get("response", "").strip()

    @property
    def config(self) -> dict:
        return {
            "provider": "ollama",
            "model": self._model,
            "base_url": self._base_url,
            "temperature": self._temperature,
        }
