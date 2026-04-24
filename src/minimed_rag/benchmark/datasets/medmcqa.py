from __future__ import annotations

from .base import BaseDataset, MCQExample

_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


class MedMCQADataset(BaseDataset):
    """Indian medical entrance (AIIMS/NEET PG) questions from medmcqa."""

    name = "medmcqa"

    def load(self, split: str = "validation") -> list[MCQExample]:
        from datasets import load_dataset  # type: ignore

        # medmcqa test split has no labels; use validation instead
        actual_split = "validation" if split == "test" else split
        ds = load_dataset("medmcqa", split=actual_split)
        examples: list[MCQExample] = []
        for i, row in enumerate(ds):
            cop: int = row["cop"]  # correct option index 0-3
            options = {
                "A": row["opa"],
                "B": row["opb"],
                "C": row["opc"],
                "D": row["opd"],
            }
            examples.append(
                MCQExample(
                    id=f"medmcqa_{row.get('id', i)}",
                    question=row["question"],
                    options=options,
                    answer=_IDX_TO_LETTER[cop],
                    dataset=self.name,
                )
            )
        return examples
