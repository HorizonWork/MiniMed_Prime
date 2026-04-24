from __future__ import annotations

import json
from pathlib import Path

from .base import BaseDataset, MCQExample

# Official BioASQ yes/no data is not freely available on HuggingFace.
# Download Task B data from https://bioasq.org (free registration required),
# then convert with scripts/convert_bioasq_yesno.py and place the output at:
_DEFAULT_LOCAL_PATH = Path("data/benchmark/bioasq_yesno.jsonl")

# Expected JSONL schema per line:
#   {"id": "...", "question": "...", "answer": "yes" | "no"}

_YESNO_OPTIONS = {"yes": "yes", "no": "no"}


class BioASQDataset(BaseDataset):
    """BioASQ yes/no questions — loaded from a local JSONL file.

    To prepare the data:
      1. Register at https://bioasq.org and download Task B JSON files.
      2. Run:  uv run python scripts/convert_bioasq_yesno.py <bioasq_taskb.json>
         This writes data/benchmark/bioasq_yesno.jsonl.
    """

    name = "bioasq_yn"

    def __init__(self, local_path: str | Path | None = None) -> None:
        self._path = Path(local_path) if local_path else _DEFAULT_LOCAL_PATH

    def load(self, split: str = "train") -> list[MCQExample]:
        if not self._path.exists():
            raise FileNotFoundError(
                f"BioASQ local file not found: {self._path}\n"
                "Register at https://bioasq.org, download Task B data, then run:\n"
                "  uv run python scripts/convert_bioasq_yesno.py <bioasq_taskb.json>"
            )

        examples: list[MCQExample] = []
        with self._path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                row = json.loads(line.strip())
                answer = row["answer"].lower().strip()
                if answer not in _YESNO_OPTIONS:
                    continue
                examples.append(
                    MCQExample(
                        id=row.get("id", f"bioasq_{i}"),
                        question=row["question"],
                        options=dict(_YESNO_OPTIONS),
                        answer=answer,
                        dataset=self.name,
                        context=row.get("context"),
                    )
                )
        return examples
