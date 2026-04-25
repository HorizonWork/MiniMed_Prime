from __future__ import annotations

from .base import ModelProvider


class HuggingFaceProvider(ModelProvider):
    """Local inference via a HuggingFace text-generation pipeline (requires `transformers`).

    Applies the model's chat template when present (instruction-tuned models like
    ``Qwen2.5-Instruct``, ``Llama-3.1-Instruct``, ``BioMistral``). Without the
    template, instruct models lose 10-20% accuracy on MIRAGE because the prompt
    doesn't match what they were fine-tuned on. Pass ``use_chat_template=False``
    to feed the raw prompt verbatim (correct for base/foundation models).
    """

    def __init__(
        self,
        model_id: str,
        device_map: str = "auto",
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        use_chat_template: bool = True,
        system_prompt: str | None = None,
    ) -> None:
        from transformers import AutoTokenizer, pipeline  # type: ignore

        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._use_chat_template = use_chat_template
        self._system_prompt = system_prompt
        self.name = f"hf:{model_id}"

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._has_chat_template = bool(getattr(self._tokenizer, "chat_template", None))

        self._pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=self._tokenizer,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )

    def _wrap_with_chat_template(self, prompt: str) -> str:
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        if self._use_chat_template and self._has_chat_template:
            wrapped = self._wrap_with_chat_template(prompt)
        else:
            wrapped = prompt

        result = self._pipe(wrapped, max_new_tokens=max_tokens)
        generated: str = result[0]["generated_text"]
        # text-generation pipelines echo the (wrapped) prompt; strip it
        if generated.startswith(wrapped):
            generated = generated[len(wrapped) :]
        return generated.strip()

    @property
    def config(self) -> dict:
        return {
            "provider": "hf",
            "model_id": self._model_id,
            "max_new_tokens": self._max_new_tokens,
            "use_chat_template": self._use_chat_template,
            "has_chat_template": self._has_chat_template,
        }
