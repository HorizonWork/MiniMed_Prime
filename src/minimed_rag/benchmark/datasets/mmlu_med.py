from __future__ import annotations

from .base import BaseDataset, MCQExample

MEDICAL_SUBSETS = [
    "clinical_knowledge",
    "medical_genetics",
    "anatomy",
    "professional_medicine",
    "college_biology",
    "college_medicine",
]

_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F"}


class MMLUMedDataset(BaseDataset):
    name = "mmlu_med"

    def __init__(self, subsets: list[str] | None = None) -> None:
        self.subsets = subsets or MEDICAL_SUBSETS

    def load(self, split: str = "test") -> list[MCQExample]:
        from datasets import load_dataset  # type: ignore

        examples: list[MCQExample] = []
        for subset in self.subsets:
            ds = load_dataset("cais/mmlu", subset, split=split)
            for i, row in enumerate(ds):
                choices: list[str] = row["choices"]
                options = {_IDX_TO_LETTER[j]: choices[j] for j in range(len(choices))}
                answer_idx: int = row["answer"]
                examples.append(
                    MCQExample(
                        id=f"mmlu_med_{subset}_{i}",
                        question=row["question"],
                        options=options,
                        answer=_IDX_TO_LETTER[answer_idx],
                        dataset=self.name,
                    )
                )
        return examples
