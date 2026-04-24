from __future__ import annotations

from collections.abc import Iterator

import pytest

from minimed_rag.benchmark.datasets.base import BaseDataset, MCQExample


@pytest.fixture
def tiny_examples() -> list[MCQExample]:
    return [
        MCQExample(
            id="tiny-1",
            question="Which option is correct?",
            options={"A": "First", "B": "Second"},
            answer="B",
            dataset="tiny",
        )
    ]


class TinyDataset(BaseDataset):
    name = "tiny"

    def __init__(self, examples: list[MCQExample]) -> None:
        self.examples = examples
        self.seen_splits: list[str] = []

    def load(self, split: str = "test") -> list[MCQExample]:
        self.seen_splits.append(split)
        return self.examples


def test_base_dataset_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseDataset()


def test_fixture_dataset_loads_examples(tiny_examples: list[MCQExample]):
    dataset = TinyDataset(tiny_examples)

    examples = dataset.load(split="validation")

    assert dataset.name == "tiny"
    assert dataset.seen_splits == ["validation"]
    assert examples == tiny_examples
    assert examples[0].context is None


def test_fixture_dataset_is_iterable_after_load(tiny_examples: list[MCQExample]):
    dataset = TinyDataset(tiny_examples)

    loaded: Iterator[MCQExample] = iter(dataset.load())

    assert next(loaded).answer == "B"
