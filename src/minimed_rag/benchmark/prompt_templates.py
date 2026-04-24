from __future__ import annotations

from .datasets.base import MCQExample

_MCQ_TEMPLATE = """\
Answer the following medical multiple-choice question.
Respond with ONLY the letter of the correct answer (A, B, C, or D).

Question: {question}

Options:
{options_block}

Answer:"""

_YESNO_TEMPLATE = """\
Answer the following biomedical question.
Respond with ONLY one word: yes, no, or maybe.
{context_block}
Question: {question}

Answer:"""


def format_prompt(example: MCQExample) -> str:
    option_keys = set(example.options.keys())

    if option_keys <= {"yes", "no", "maybe"}:
        if example.context:
            context_block = f"\nContext:\n{example.context}\n"
        else:
            context_block = ""
        return _YESNO_TEMPLATE.format(
            context_block=context_block,
            question=example.question,
        )

    options_block = "\n".join(f"{k}. {v}" for k, v in sorted(example.options.items()))
    question = example.question
    if example.context:
        question = f"Context: {example.context}\n\n{question}"

    return _MCQ_TEMPLATE.format(question=question, options_block=options_block)
