"""Unit tests for the HF provider chat-template behavior.

Pure constructor-free unit tests — we monkeypatch ``transformers`` so the
test doesn't need the heavy dep and doesn't download a model. The intent
is to verify the prompt-wrapping logic, not the inference path.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import minimed_rag.benchmark.providers.hf_local as hf_module


class _FakeTokenizer:
    def __init__(self, with_template: bool = True):
        self.chat_template = "{% for m in messages %}{{m['content']}}{% endfor %}" if with_template else None
        self.applied_calls: list[list[dict]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.applied_calls.append(messages)
        return "<|wrapped|>" + "".join(m["content"] for m in messages) + "<|end|>"


class _FakePipe:
    def __init__(self):
        self.received: list[str] = []

    def __call__(self, prompt, max_new_tokens=64):
        self.received.append(prompt)
        return [{"generated_text": prompt + "ANSWER"}]


@pytest.fixture
def patched_hf(monkeypatch):
    """Replace ``transformers.AutoTokenizer`` and ``pipeline`` with fakes."""

    fake_tokenizer = _FakeTokenizer(with_template=True)
    fake_pipe = _FakePipe()

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id):
            return fake_tokenizer

    def _pipeline(*args, **kwargs):
        return fake_pipe

    fake_transformers = SimpleNamespace(
        AutoTokenizer=_AutoTokenizer,
        pipeline=_pipeline,
    )

    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    return fake_tokenizer, fake_pipe


def test_uses_chat_template_when_available(patched_hf):
    tokenizer, pipe = patched_hf
    provider = hf_module.HuggingFaceProvider(model_id="fake/instruct")
    out = provider.generate("Question?")

    # Should have wrapped via chat template
    assert tokenizer.applied_calls
    assert tokenizer.applied_calls[0] == [{"role": "user", "content": "Question?"}]
    assert pipe.received[0].startswith("<|wrapped|>")
    # Strip should remove the wrapped prefix and leave the suffix
    assert out == "ANSWER"


def test_skip_chat_template_when_disabled(patched_hf):
    tokenizer, pipe = patched_hf
    provider = hf_module.HuggingFaceProvider(model_id="fake/base", use_chat_template=False)
    provider.generate("raw prompt")

    assert tokenizer.applied_calls == []
    assert pipe.received[0] == "raw prompt"


def test_skip_chat_template_when_tokenizer_lacks_one(monkeypatch):
    fake_tokenizer = _FakeTokenizer(with_template=False)
    fake_pipe = _FakePipe()

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id):
            return fake_tokenizer

    def _pipeline(*args, **kwargs):
        return fake_pipe

    fake_transformers = SimpleNamespace(
        AutoTokenizer=_AutoTokenizer,
        pipeline=_pipeline,
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    provider = hf_module.HuggingFaceProvider(model_id="fake/base", use_chat_template=True)
    provider.generate("hi")
    # Even though use_chat_template=True, no template => raw prompt is fed
    assert fake_tokenizer.applied_calls == []
    assert fake_pipe.received[0] == "hi"


def test_system_prompt_included_when_set(patched_hf):
    tokenizer, _ = patched_hf
    provider = hf_module.HuggingFaceProvider(
        model_id="fake/instruct", system_prompt="You are helpful."
    )
    provider.generate("Hello")
    assert tokenizer.applied_calls[0][0] == {"role": "system", "content": "You are helpful."}
    assert tokenizer.applied_calls[0][1] == {"role": "user", "content": "Hello"}


def test_config_reports_template_status(patched_hf):
    provider = hf_module.HuggingFaceProvider(model_id="fake/instruct")
    cfg = provider.config
    assert cfg["provider"] == "hf"
    assert cfg["model_id"] == "fake/instruct"
    assert cfg["use_chat_template"] is True
    assert cfg["has_chat_template"] is True
