"""Convert official BioASQ Task B JSON → data/benchmark/bioasq_yesno.jsonl.

Usage:
    uv run python scripts/convert_bioasq_yesno.py path/to/BioASQ-taskB.json

The Task B JSON has a "questions" list. We extract only type=="yesno" items and
write one line per question:
    {"id": "...", "question": "...", "answer": "yes" | "no"}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT_PATH = Path("data/benchmark/bioasq_yesno.jsonl")


def convert(src: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    questions = data.get("questions", [])

    yesno = [q for q in questions if q.get("type") == "yesno"]
    if not yesno:
        print(f"No yes/no questions found in {src}")
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for q in yesno:
            exact = q.get("exact_answer", "")
            answer = (exact[0] if isinstance(exact, list) else exact).lower().strip()
            if answer not in {"yes", "no"}:
                continue
            f.write(
                json.dumps(
                    {"id": q["id"], "question": q["body"], "answer": answer},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"Wrote {sum(1 for _ in OUT_PATH.open())} yes/no questions → {OUT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    convert(Path(sys.argv[1]))
