from __future__ import annotations

from .base import BaseDataset, MCQExample


class MedQADataset(BaseDataset):
    """4-option USMLE questions from GBaker/MedQA-USMLE-4-options."""

    name = "medqa_us"

    def load(self, split: str = "test") -> list[MCQExample]:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
        examples: list[MCQExample] = []
        for i, row in enumerate(ds):
            options: dict[str, str] = row["options"]
            answer_idx: str = row["answer_idx"]
            examples.append(
                MCQExample(
                    id=f"medqa_us_{i}",
                    question=row["question"],
                    options=options,
                    answer=answer_idx.upper(),
                    dataset=self.name,
                )
            )
        return examples
