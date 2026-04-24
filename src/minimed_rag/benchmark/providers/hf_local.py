from __future__ import annotations

from .base import ModelProvider


class HuggingFaceProvider(ModelProvider):
    """Local inference via a HuggingFace text-generation pipeline (requires `transformers`)."""

    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> None:
        from transformers import pipeline  # type: ignore

        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self.name = f"hf:{model_id}"
        self._pipe = pipeline(
            "text-generation",
            model=model_id,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        result = self._pipe(prompt, max_new_tokens=max_tokens)
        generated: str = result[0]["generated_text"]
        # Strip the prompt prefix that text-generation pipelines echo back
        if generated.startswith(prompt):
            generated = generated[len(prompt) :]
        return generated.strip()

    @property
    def config(self) -> dict:
        return {
            "provider": "hf",
            "model_id": self._model_id,
            "max_new_tokens": self._max_new_tokens,
        }
