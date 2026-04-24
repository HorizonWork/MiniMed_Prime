from __future__ import annotations

import re


def parse_mcq_answer(text: str, valid_options: set[str]) -> str | None:
    """Extract a multiple-choice letter from model output."""
    text = text.strip()

    # 1. Entire output is a single valid letter
    if text.upper() in valid_options:
        return text.upper()

    # 2. Explicit "Answer: X" or "The answer is X"
    m = re.search(r"(?:answer|option)[:\s]+\(?([A-E])\)?", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid_options:
        return m.group(1).upper()

    # 3. Parenthesised letter "(X)"
    m = re.search(r"\(([A-E])\)", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid_options:
        return m.group(1).upper()

    # 4. First non-empty line contains a lone valid letter
    for line in text.splitlines():
        stripped = line.strip().upper()
        if stripped in valid_options:
            return stripped

    # 5. Letter followed by ". " or ") " (option-list style)
    m = re.search(r"\b([A-E])[.)]\s", text)
    if m and m.group(1).upper() in valid_options:
        return m.group(1).upper()

    # 6. Any isolated letter that is a valid option (first match wins)
    for m in re.finditer(r"\b([A-E])\b", text, re.IGNORECASE):
        letter = m.group(1).upper()
        if letter in valid_options:
            return letter

    return None


def parse_yesno_answer(text: str) -> str | None:
    """Extract yes / no / maybe from model output."""
    lower = text.strip().lower()

    if lower in {"yes", "no", "maybe"}:
        return lower

    for word in ("yes", "no", "maybe"):
        if re.search(rf"\b{word}\b", lower):
            return word

    return None


def parse_answer(text: str, options: dict[str, str]) -> str | None:
    """Route to the appropriate parser based on the option key set."""
    option_keys = set(options.keys())
    if option_keys <= {"yes", "no", "maybe"}:
        return parse_yesno_answer(text)
    return parse_mcq_answer(text, option_keys)
