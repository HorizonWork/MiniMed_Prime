from __future__ import annotations

from .base import BaseDataset, MCQExample

_YESNO_OPTIONS = {"yes": "yes", "no": "no", "maybe": "maybe"}


class PubMedQADataset(BaseDataset):
    """PubMedQA labeled yes/no/maybe from qiaojin/PubMedQA (pqa_labeled)."""

    name = "pubmedqa"

    def load(self, split: str = "train") -> list[MCQExample]:
        from datasets import load_dataset  # type: ignore

        # pqa_labeled only has a train split
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        examples: list[MCQExample] = []
        for row in ds:
            ctx = row.get("context", {})
            ctx_texts: list[str] = ctx.get("contexts", []) if isinstance(ctx, dict) else []
            context = " ".join(ctx_texts) if ctx_texts else None

            decision: str = row["final_decision"].lower().strip()
            if decision not in _YESNO_OPTIONS:
                continue

            examples.append(
                MCQExample(
                    id=f"pubmedqa_{row['pubid']}",
                    question=row["question"],
                    options=dict(_YESNO_OPTIONS),
                    answer=decision,
                    dataset=self.name,
                    context=context,
                )
            )
        return examples
