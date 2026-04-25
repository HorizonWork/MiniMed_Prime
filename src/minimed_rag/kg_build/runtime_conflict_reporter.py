"""Runtime graph-vs-text conflict reporter.

Sibling to ``conflict_detector.py`` — that module checks the *offline* KG
for self-conflicting assertions before serving. This module runs *during*
RAG inference: it cross-checks the KG facts retrieved for a question
against the text chunks retrieved for the same question and flags
contradictions (negation, polarity flip).

Detection here is intentionally heuristic — string-pattern matching plus
a small predicate-verb lookup. NLI-grade detection is Phase 6+ scope; the
``Conflict`` dataclass and ``write_conflicts_jsonl`` helper are stable
contracts so a smarter detector can drop in without touching the eval
loop or the consumers.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minimed_rag.retrieval.bm25_retriever import RetrievalResult
    from minimed_rag.retrieval.graph_retriever import GraphFact, GraphPath


# Map predicate name → list of verb stems used in negation patterns.
PREDICATE_VERB_MAP: dict[str, tuple[str, ...]] = {
    "treats": ("treat", "cure", "heal"),
    "causes_adverse_event": ("cause", "induce", "trigger"),
    "causes": ("cause", "induce", "trigger"),
    "prevents": ("prevent", "block"),
    "targets": ("target", "bind", "inhibit"),
    "associated_with": ("associated", "linked", "correlated"),
    "increases_risk_of": ("increase", "raise", "elevate"),
    "decreases_risk_of": ("decrease", "lower", "reduce"),
    "manifested_as": ("manifest", "present"),
    "participates_in": ("participate", "involved"),
    "involved_in": ("involve", "participate"),
    "gene_associated_with_disease": ("associated", "linked"),
}

# Map predicate → predicates that flip polarity (graph says X, text says Y → conflict).
PREDICATE_OPPOSITE_MAP: dict[str, tuple[str, ...]] = {
    "treats": ("worsen", "aggravate", "exacerbate"),
    "prevents": ("cause", "induce", "trigger"),
    "causes": ("prevent", "block"),
    "increases_risk_of": ("decrease", "lower", "reduce", "protect"),
    "decreases_risk_of": ("increase", "raise", "elevate"),
}

NEGATION_TOKENS = (
    r"\b(?:no|not|does\s+not|doesn['’]t|never|fails?\s+to|did\s+not|cannot|can['’]t)\b"
)


@dataclass(slots=True)
class Conflict:
    """Single graph-vs-text contradiction."""

    fact_assertion_id: str
    fact_subject: str
    fact_predicate: str
    fact_object: str
    fact_confidence: float
    fact_source_system: str
    chunk_id: str
    chunk_source: str
    chunk_excerpt: str
    rule: str          # which detector rule fired
    severity: str      # "negation" | "polarity_flip"


def _verbs_for(predicate: str) -> tuple[str, ...]:
    return PREDICATE_VERB_MAP.get(predicate, ())


def _opposites_for(predicate: str) -> tuple[str, ...]:
    return PREDICATE_OPPOSITE_MAP.get(predicate, ())


def _excerpt(text: str, idx: int, radius: int = 150) -> str:
    start = max(0, idx - radius)
    end = min(len(text), idx + radius)
    snippet = text[start:end].strip()
    return snippet


def _find_window_with_both(text: str, subject: str, obj: str, window: int = 200) -> int | None:
    """Return position in ``text`` near where ``subject`` and ``obj`` co-occur.

    Returns the midpoint between the two mentions if both appear within
    ``window`` chars; otherwise ``None``.
    """
    if not subject or not obj:
        return None
    text_lower = text.lower()
    s_idx = text_lower.find(subject.lower())
    o_idx = text_lower.find(obj.lower())
    if s_idx < 0 or o_idx < 0:
        return None
    if abs(s_idx - o_idx) > window:
        return None
    return (s_idx + o_idx) // 2


def _matches_negation(window_text: str, verbs: tuple[str, ...]) -> str | None:
    """Return the matched verb if any negation+verb pattern fires inside the window."""
    if not verbs:
        return None
    verbs_pattern = "|".join(re.escape(v) for v in verbs)
    pattern = rf"(?i){NEGATION_TOKENS}\s+(?:\w+\s+){{0,3}}(?:{verbs_pattern})\w*"
    match = re.search(pattern, window_text)
    if match:
        for verb in verbs:
            if verb in match.group(0).lower():
                return verb
    return None


def _matches_opposite(window_text: str, opposites: tuple[str, ...]) -> str | None:
    """Return the matched opposite verb if a polarity flip phrase fires."""
    if not opposites:
        return None
    pattern = rf"(?i)\b(?:{'|'.join(re.escape(v) for v in opposites)})\w*"
    match = re.search(pattern, window_text)
    if match:
        return match.group(0)
    return None


def _matches_affirmative(window_text: str, verbs: tuple[str, ...]) -> str | None:
    """Return a predicate verb when text affirmatively states the graph predicate."""
    if not verbs:
        return None
    pattern = rf"(?i)\b(?:{'|'.join(re.escape(v) for v in verbs)})\w*"
    match = re.search(pattern, window_text)
    if match:
        return match.group(0)
    return None


class RuntimeConflictReporter:
    """Cross-check retrieved graph paths against retrieved text chunks."""

    def __init__(
        self,
        *,
        co_occurrence_window: int = 200,
        excerpt_radius: int = 150,
    ) -> None:
        self.co_occurrence_window = co_occurrence_window
        self.excerpt_radius = excerpt_radius

    def find_conflicts(
        self,
        graph_paths: list[GraphPath],
        text_results: list[RetrievalResult] | None,
    ) -> list[Conflict]:
        """Return all conflict records discovered between paths and text chunks."""
        if not graph_paths or not text_results:
            return []

        conflicts: list[Conflict] = []
        for path in graph_paths:
            for hop in path.hops:
                for result in text_results:
                    chunk = getattr(result, "chunk", None)
                    if chunk is None:
                        continue
                    text = getattr(chunk, "text", "") or ""
                    if not text:
                        continue
                    conflict = self._check_fact_against_chunk(hop, chunk, text)
                    if conflict is not None:
                        conflicts.append(conflict)
        return conflicts

    def _check_fact_against_chunk(
        self,
        fact: GraphFact,
        chunk,
        text: str,
    ) -> Conflict | None:
        midpoint = _find_window_with_both(
            text, fact.subject_name, fact.object_name, window=self.co_occurrence_window
        )
        if midpoint is None:
            return None

        excerpt = _excerpt(text, midpoint, radius=self.excerpt_radius)

        verbs = _verbs_for(fact.predicate)
        if verb := _matches_negation(excerpt, verbs):
            if fact.polarity == "negative":
                return None
            return self._build_conflict(
                fact, chunk, excerpt, rule=f"negation:{verb}", severity="negation"
            )

        opposites = _opposites_for(fact.predicate)
        if op := _matches_opposite(excerpt, opposites):
            if fact.polarity == "negative":
                return None
            return self._build_conflict(
                fact, chunk, excerpt, rule=f"opposite:{op}", severity="polarity_flip"
            )

        if fact.polarity == "negative":
            if verb := _matches_affirmative(excerpt, verbs):
                return self._build_conflict(
                    fact,
                    chunk,
                    excerpt,
                    rule=f"affirmation:{verb}",
                    severity="polarity_flip",
                )

        return None

    @staticmethod
    def _build_conflict(fact, chunk, excerpt: str, *, rule: str, severity: str) -> Conflict:
        return Conflict(
            fact_assertion_id=fact.assertion_id,
            fact_subject=fact.subject_name,
            fact_predicate=fact.predicate,
            fact_object=fact.object_name,
            fact_confidence=fact.confidence,
            fact_source_system=fact.source_system,
            chunk_id=str(getattr(chunk, "id", "") or ""),
            chunk_source=str(getattr(chunk, "source", "") or ""),
            chunk_excerpt=excerpt,
            rule=rule,
            severity=severity,
        )


def write_conflicts_jsonl(
    conflicts: list[Conflict],
    output_dir: str | Path,
    suite: str,
    timestamp: str,
) -> Path:
    """Append all conflicts to ``artifacts/reports/conflicts_<suite>_<ts>.jsonl``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_ts = timestamp.replace(":", "-").replace(".", "-")
    path = out / f"conflicts_{suite}_{safe_ts}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for conflict in conflicts:
            fh.write(json.dumps(asdict(conflict)) + "\n")
    return path


__all__ = [
    "Conflict",
    "RuntimeConflictReporter",
    "write_conflicts_jsonl",
    "PREDICATE_VERB_MAP",
    "PREDICATE_OPPOSITE_MAP",
]
