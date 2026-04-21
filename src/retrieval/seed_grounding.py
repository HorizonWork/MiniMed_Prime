from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence, TypeVar

from pydantic import BaseModel, Field

from src.layers.layer4_judges import JudgeBase
from src.retrieval.primekg_grounding import (
    NODE_TYPE_TO_ENTITY_TYPE,
    node_lookup_keys,
    normalize_basic_text,
    normalize_lookup_text,
    parse_node_cuis,
    resolve_primekg_node_identifier,
)
from src.schemas import QuestionEntity
from src.utils.kaggle_env import KaggleEnv

T = TypeVar("T")

DEFAULT_GROUNDING_TOP_K = 20
DEFAULT_EXACT_KEY_SCORE = 1.00
DEFAULT_ALIAS_KEY_SCORE = 0.92
DEFAULT_SUFFIX_KEY_SCORE = 0.92
DEFAULT_TOKEN_CONTAINMENT_SCORE = 0.85
DEFAULT_TYPE_BONUS = 0.05
DEFAULT_AUTO_LINK_THRESHOLD = 0.90
DEFAULT_MARGIN_LINK_THRESHOLD = 0.84
DEFAULT_MARGIN_GAP = 0.08
DEFAULT_MIN_FUZZY_SCORE = 0.60
DEFAULT_MIN_TOKEN_LENGTH = 3
DEFAULT_MAX_FUZZY_CANDIDATES = 200
DEFAULT_MAX_EXTRACTED_NGRAM = 6
DEFAULT_OPENAI_GROUNDER = "gpt-4o-mini"
DEFAULT_GEMINI_GROUNDER = "gemini-2.5-flash"

GROUNDING_MODES = ("baseline_current", "deterministic_v2", "llm_assisted_v1")
SEED_EXTRACTOR_BACKEND = "lightweight_seed_rules"
QUESTION_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
OPTION_LINE_PATTERN = re.compile(r"(?mi)^\s*(?:[A-H][\.\)])\s*(.+?)\s*$")
STRUCTURAL_SEGMENT_PATTERN = re.compile(r"(?i)\b(?:options|answer choices?)\b\s*:?\s*")
SEED_EXTRACTION_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "answer",
        "answers",
        "choices",
        "except",
        "following",
        "for",
        "from",
        "in",
        "is",
        "made",
        "of",
        "option",
        "options",
        "or",
        "the",
        "to",
        "up",
        "with",
    }
)
ANATOMY_HINT_TOKENS = frozenset(
    {
        "anatomy",
        "artery",
        "diaphragm",
        "duct",
        "fascia",
        "gland",
        "ligament",
        "membrane",
        "muscle",
        "nerve",
        "perineal",
        "perineus",
        "sphincter",
        "structure",
        "urethrae",
        "vein",
    }
)
GENERIC_ANATOMY_SINGLETONS = frozenset(
    {
        "artery",
        "diaphragm",
        "duct",
        "fascia",
        "gland",
        "ligament",
        "membrane",
        "muscle",
        "nerve",
        "sphincter",
        "structure",
        "vein",
    }
)
ANATOMY_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "deep transverse perineus": ("perineal muscle",),
    "deep transverse perineal": ("perineal muscle",),
    "perinial membrane": ("perineal membrane",),
    "sphincter urethrae": ("urethral sphincter",),
}
ANATOMY_QUERY_TOKEN_ALIASES = {
    "perinial": "perineal",
    "perineus": "perineal",
    "urethrae": "urethral",
}


@dataclass(slots=True)
class PrimeKGNodeRecord:
    node_id: str
    node_name: str
    node_type: str
    source: str
    exact_key: str
    lookup_key: str
    suffix_keys: tuple[str, ...]
    tokens: tuple[str, ...]
    cuis: tuple[str, ...] = ()


@dataclass(slots=True)
class EntityGroundingCandidate:
    node_id: str
    node_name: str
    node_type: str
    source: str
    match_features: list[str]
    score: float


@dataclass(slots=True)
class EntityGroundingDecision:
    surface: str
    entity_type: str
    cui: str | None
    linked_node_id: str | None
    linked_node_name: str | None
    backend_used: str
    decision_reason: str
    candidates: list[EntityGroundingCandidate] = field(default_factory=list)


class SeedGrounder(Protocol):
    backend_used: str
    llm_used: bool
    fallback_reason: str | None

    def ground(self, question: str, entities: Sequence[QuestionEntity]) -> list[EntityGroundingDecision]:
        raise NotImplementedError


class SeedEntityExtractor(Protocol):
    backend_used: str

    def extract(self, question: str, question_type_hint: str | None = None) -> list[QuestionEntity]:
        raise NotImplementedError


class PrimeKGNodeCatalog:
    def __init__(self, nodes_csv: Path) -> None:
        resolved_nodes_csv = nodes_csv if nodes_csv.is_absolute() else KaggleEnv.path(nodes_csv)
        self.nodes_csv = resolved_nodes_csv
        self.records_by_id: dict[str, PrimeKGNodeRecord] = {}
        self.exact_index: dict[str, set[str]] = defaultdict(set)
        self.alias_index: dict[str, set[str]] = defaultdict(set)
        self.suffix_index: dict[str, set[str]] = defaultdict(set)
        self.cui_index: dict[str, set[str]] = defaultdict(set)
        self.token_index: dict[str, set[str]] = defaultdict(set)
        self.prefix_index: dict[str, set[str]] = defaultdict(set)
        self._load()

    @classmethod
    def from_primekg_path(cls, primekg_path: str | Path) -> PrimeKGNodeCatalog:
        resolved_primekg_path = primekg_path if isinstance(primekg_path, Path) else Path(primekg_path)
        resolved_primekg_path = (
            resolved_primekg_path
            if resolved_primekg_path.is_absolute()
            else KaggleEnv.path(resolved_primekg_path)
        )
        if resolved_primekg_path.is_dir():
            nodes_csv = resolved_primekg_path / "nodes.csv"
        else:
            nodes_csv = resolved_primekg_path.parent / "nodes.csv"
        if not nodes_csv.exists():
            raise FileNotFoundError(
                f"PrimeKG data-only grounding requires nodes.csv, but none was found at {nodes_csv}."
            )
        return cls(nodes_csv=nodes_csv)

    def _load(self) -> None:
        if not self.nodes_csv.exists():
            raise FileNotFoundError(f"PrimeKG nodes.csv was not found at {self.nodes_csv}.")

        with self.nodes_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            node_id_column = _first_present(header, ("node_id", "id", "primekg_node_id"))
            node_name_column = _first_present(header, ("node_name", "name"))
            node_type_column = _first_present(header, ("node_type", "type"))
            node_source_column = _first_present(header, ("node_source", "source"))
            node_cui_column = _first_present(header, ("node_cui", "cui", "umls_cui", "umls_id"))
            if node_id_column is None or node_name_column is None or node_type_column is None:
                raise ValueError("PrimeKG nodes.csv is missing required node_id, node_name, or node_type columns.")
            node_index_column = _first_present(header, ("node_index", "index", "id"))
            if node_index_column is None:
                raise ValueError("PrimeKG nodes.csv is missing required node_index column for stable data-only grounding.")
            rows = list(reader)
            duplicate_node_ids = {
                raw_node_id
                for raw_node_id, count in Counter(
                    str(row.get(node_id_column, "") or "").strip() or f"node:{str(row.get(node_index_column, '') or '').strip()}"
                    for row in rows
                ).items()
                if count > 1
            }

            for row in rows:
                node_index = str(row.get(node_index_column, "") or "").strip()
                raw_node_id = str(row.get(node_id_column, "") or "").strip()
                node_id = resolve_primekg_node_identifier(
                    node_index=node_index,
                    node_id=raw_node_id,
                    duplicate_node_ids=duplicate_node_ids,
                )
                node_name = str(row.get(node_name_column, "") or "").strip()
                node_type = str(row.get(node_type_column, "") or "other").strip() or "other"
                source = str(row.get(node_source_column, "") or "primekg").strip() or "primekg"
                if not node_id or not node_name:
                    continue
                exact_key = normalize_basic_text(node_name)
                lookup_key = normalize_lookup_text(node_name)
                extra_keys = tuple(
                    key
                    for key in sorted(node_lookup_keys(node_name))
                    if key and key not in {exact_key, lookup_key}
                )
                tokens = tuple(token for token in lookup_key.split() if len(token) >= DEFAULT_MIN_TOKEN_LENGTH)
                cuis = tuple(parse_node_cuis(str(row.get(node_cui_column, "") or ""))) if node_cui_column else ()
                record = PrimeKGNodeRecord(
                    node_id=node_id,
                    node_name=node_name,
                    node_type=node_type,
                    source=source,
                    exact_key=exact_key,
                    lookup_key=lookup_key,
                    suffix_keys=extra_keys,
                    tokens=tokens,
                    cuis=cuis,
                )
                self.records_by_id[node_id] = record
                if exact_key:
                    self.exact_index[exact_key].add(node_id)
                if lookup_key:
                    self.alias_index[lookup_key].add(node_id)
                    self.prefix_index[lookup_key[:3]].add(node_id)
                for key in extra_keys:
                    self.suffix_index[key].add(node_id)
                for token in tokens:
                    self.token_index[token].add(node_id)
                    self.prefix_index[token[:3]].add(node_id)
                for cui in cuis:
                    self.cui_index[cui].add(node_id)

    def records_for_ids(self, node_ids: Sequence[str]) -> list[PrimeKGNodeRecord]:
        return [self.records_by_id[node_id] for node_id in node_ids if node_id in self.records_by_id]

    def fuzzy_pool(self, query: str) -> set[str]:
        query_lookup = normalize_lookup_text(query)
        query_tokens = [token for token in query_lookup.split() if len(token) >= DEFAULT_MIN_TOKEN_LENGTH]
        candidate_ids: set[str] = set()
        for token in query_tokens:
            candidate_ids.update(self.token_index.get(token, set()))
            candidate_ids.update(self.prefix_index.get(token[:3], set()))
        if query_lookup:
            candidate_ids.update(self.prefix_index.get(query_lookup[:3], set()))
        if len(candidate_ids) <= DEFAULT_MAX_FUZZY_CANDIDATES:
            return candidate_ids
        ranked_candidates = sorted(
            candidate_ids,
            key=lambda node_id: (
                -_token_overlap_count(query_tokens, self.records_by_id[node_id].tokens),
                self.records_by_id[node_id].node_name,
            ),
        )
        return set(ranked_candidates[:DEFAULT_MAX_FUZZY_CANDIDATES])


class LightweightSeedEntityExtractor:
    def __init__(self, catalog: PrimeKGNodeCatalog, *, max_ngram: int = DEFAULT_MAX_EXTRACTED_NGRAM) -> None:
        self.catalog = catalog
        self.max_ngram = max(1, max_ngram)
        self.backend_used = SEED_EXTRACTOR_BACKEND

    def extract(self, question: str, question_type_hint: str | None = None) -> list[QuestionEntity]:
        del question_type_hint
        extracted: list[QuestionEntity] = []
        seen_keys: set[tuple[str, str]] = set()
        for segment in self._raw_surface_segments(question):
            self._append_entity_if_new(
                extracted=extracted,
                seen_keys=seen_keys,
                surface=segment,
                entity_type=self._infer_surface_entity_type(segment),
            )
        for surface, entity_type in self._matched_spans(question):
            self._append_entity_if_new(
                extracted=extracted,
                seen_keys=seen_keys,
                surface=surface,
                entity_type=entity_type,
            )
        return extracted

    def _segment_lines(self, question: str) -> tuple[list[str], list[str]]:
        cleaned_question = STRUCTURAL_SEGMENT_PATTERN.sub("\n", question or "")
        option_segments = [
            _clean_seed_surface(match.group(1))
            for match in OPTION_LINE_PATTERN.finditer(cleaned_question)
        ]
        stem_lines: list[str] = []
        for raw_line in cleaned_question.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if OPTION_LINE_PATTERN.match(line):
                continue
            stem_lines.append(line)
        return stem_lines, option_segments

    def _raw_surface_segments(self, question: str) -> list[str]:
        stem_lines, option_segments = self._segment_lines(question)
        raw_segments = [segment for segment in option_segments if _is_seed_surface_candidate(segment)]
        raw_segments.extend(
            segment
            for segment in stem_lines
            if _is_seed_surface_candidate(segment) and len(QUESTION_TOKEN_PATTERN.findall(segment)) <= 4
        )
        return _dedupe_preserve_order(raw_segments)

    def _matched_spans(self, question: str) -> list[tuple[str, str]]:
        matches: list[tuple[str, str]] = []
        stem_lines, _ = self._segment_lines(question)
        for segment in stem_lines:
            tokens = QUESTION_TOKEN_PATTERN.findall(segment)
            if not tokens:
                continue
            max_window = min(self.max_ngram, len(tokens))
            for window in range(max_window, 0, -1):
                for start in range(0, len(tokens) - window + 1):
                    span_tokens = tokens[start : start + window]
                    span_surface = " ".join(span_tokens).strip()
                    if not _is_seed_surface_candidate(span_surface):
                        continue
                    matched_type = self._matched_entity_type(span_surface)
                    if matched_type is not None:
                        if (
                            len(span_tokens) == 1
                            and matched_type == "anatomy"
                            and normalize_lookup_text(span_surface) in GENERIC_ANATOMY_SINGLETONS
                        ):
                            continue
                        matches.append((span_surface, matched_type))
        return _dedupe_preserve_order(matches)

    def _matched_entity_type(self, surface: str) -> str | None:
        query_basic = normalize_basic_text(surface)
        query_lookup = normalize_lookup_text(surface)
        exact_or_alias_ids = set(self.catalog.exact_index.get(query_basic, set())) | set(
            self.catalog.alias_index.get(query_lookup, set())
        )
        matched_node_ids = exact_or_alias_ids or set(self.catalog.suffix_index.get(query_lookup, set()))
        if not matched_node_ids:
            return None
        node_types = {
            NODE_TYPE_TO_ENTITY_TYPE.get(normalize_basic_text(self.catalog.records_by_id[node_id].node_type), "other")
            for node_id in matched_node_ids
            if node_id in self.catalog.records_by_id
        }
        if len(node_types) == 1:
            return next(iter(node_types))
        if "other" in node_types and len(node_types) == 2:
            return next(iter(node_types - {"other"}), "other")
        return "other"

    def _infer_surface_entity_type(self, surface: str) -> str:
        matched_type = self._matched_entity_type(surface)
        if matched_type is not None:
            return matched_type
        normalized = normalize_lookup_text(surface)
        surface_tokens = set(normalized.split())
        if surface_tokens & ANATOMY_HINT_TOKENS:
            return "anatomy"
        return "other"

    def _append_entity_if_new(
        self,
        *,
        extracted: list[QuestionEntity],
        seen_keys: set[tuple[str, str]],
        surface: str,
        entity_type: str,
    ) -> None:
        normalized_surface = normalize_basic_text(surface)
        if not normalized_surface:
            return
        key = (normalized_surface, entity_type)
        if key in seen_keys:
            return
        seen_keys.add(key)
        extracted.append(
            QuestionEntity(
                surface=surface,
                cui=None,
                primekg_node_id=None,
                entity_type=entity_type,
            )
        )


class DeterministicSeedGrounder:
    def __init__(self, catalog: PrimeKGNodeCatalog, *, top_k: int = DEFAULT_GROUNDING_TOP_K) -> None:
        self.catalog = catalog
        self.top_k = top_k
        self.backend_used = "deterministic_v2"
        self.llm_used = False
        self.fallback_reason: str | None = None

    def ground(self, question: str, entities: Sequence[QuestionEntity]) -> list[EntityGroundingDecision]:
        del question
        decisions: list[EntityGroundingDecision] = []
        for entity in entities:
            decisions.append(self._ground_entity(entity))
        return decisions

    def _ground_entity(self, entity: QuestionEntity) -> EntityGroundingDecision:
        candidate_scores: dict[str, float] = {}
        candidate_features: dict[str, set[str]] = defaultdict(set)
        query_variants = _query_variants(entity)

        if entity.cui:
            for node_id in self.catalog.cui_index.get(entity.cui.lower(), set()):
                self._register_candidate(
                    entity=entity,
                    node_id=node_id,
                    base_score=DEFAULT_EXACT_KEY_SCORE,
                    feature="cui_hit",
                    candidate_scores=candidate_scores,
                    candidate_features=candidate_features,
                )

        for query_basic, query_lookup, query_feature in query_variants:
            exact_hits = set(self.catalog.exact_index.get(query_basic, set()))
            for node_id in exact_hits:
                self._register_candidate(
                    entity=entity,
                    node_id=node_id,
                    base_score=DEFAULT_EXACT_KEY_SCORE,
                    feature="exact_key_hit",
                    candidate_scores=candidate_scores,
                    candidate_features=candidate_features,
                    extra_feature=query_feature,
                )

            for node_id in self.catalog.alias_index.get(query_lookup, set()):
                if node_id in exact_hits:
                    continue
                self._register_candidate(
                    entity=entity,
                    node_id=node_id,
                    base_score=DEFAULT_ALIAS_KEY_SCORE,
                    feature="alias_key_hit",
                    candidate_scores=candidate_scores,
                    candidate_features=candidate_features,
                    extra_feature=query_feature,
                )

            for node_id in self.catalog.suffix_index.get(query_lookup, set()):
                self._register_candidate(
                    entity=entity,
                    node_id=node_id,
                    base_score=DEFAULT_SUFFIX_KEY_SCORE,
                    feature="suffix_key_hit",
                    candidate_scores=candidate_scores,
                    candidate_features=candidate_features,
                    extra_feature=query_feature,
                )

        current_best = max(candidate_scores.values(), default=0.0)
        if current_best < DEFAULT_ALIAS_KEY_SCORE:
            fuzzy_candidates: set[str] = set()
            for _, query_lookup, _ in query_variants:
                fuzzy_candidates.update(self.catalog.fuzzy_pool(query_lookup))
            for node_id in fuzzy_candidates:
                record = self.catalog.records_by_id[node_id]
                for _, query_lookup, query_feature in query_variants:
                    token_containment = _token_containment_score(query_lookup, record.lookup_key)
                    fuzzy_score = _fuzzy_score(query_lookup, record.lookup_key)
                    if token_containment > 0.0:
                        self._register_candidate(
                            entity=entity,
                            node_id=node_id,
                            base_score=token_containment,
                            feature="token_containment",
                            candidate_scores=candidate_scores,
                            candidate_features=candidate_features,
                            extra_feature=query_feature,
                        )
                    if fuzzy_score >= DEFAULT_MIN_FUZZY_SCORE:
                        self._register_candidate(
                            entity=entity,
                            node_id=node_id,
                            base_score=fuzzy_score,
                            feature="fuzzy_similarity",
                            candidate_scores=candidate_scores,
                            candidate_features=candidate_features,
                            extra_feature=query_feature,
                        )

        ranked_candidates = self._rank_candidates(candidate_scores, candidate_features)
        linked_node_id: str | None = None
        linked_node_name: str | None = None
        decision_reason = "no_candidates"
        if ranked_candidates:
            top_candidate = ranked_candidates[0]
            runner_up_score = ranked_candidates[1].score if len(ranked_candidates) > 1 else 0.0
            margin = top_candidate.score - runner_up_score
            if top_candidate.score >= DEFAULT_AUTO_LINK_THRESHOLD:
                linked_node_id = top_candidate.node_id
                linked_node_name = top_candidate.node_name
                decision_reason = "auto_link_high_confidence"
            elif top_candidate.score >= DEFAULT_MARGIN_LINK_THRESHOLD and margin >= DEFAULT_MARGIN_GAP:
                linked_node_id = top_candidate.node_id
                linked_node_name = top_candidate.node_name
                decision_reason = "auto_link_margin"
            else:
                decision_reason = "unresolved_below_threshold"

        return EntityGroundingDecision(
            surface=entity.surface,
            entity_type=entity.entity_type,
            cui=entity.cui,
            linked_node_id=linked_node_id,
            linked_node_name=linked_node_name,
            backend_used=self.backend_used,
            decision_reason=decision_reason,
            candidates=ranked_candidates[: self.top_k],
        )

    def _register_candidate(
        self,
        *,
        entity: QuestionEntity,
        node_id: str,
        base_score: float,
        feature: str,
        candidate_scores: dict[str, float],
        candidate_features: dict[str, set[str]],
        extra_feature: str | None = None,
    ) -> None:
        record = self.catalog.records_by_id[node_id]
        type_bonus = DEFAULT_TYPE_BONUS if _entity_matches_node_type(entity.entity_type, record.node_type) else 0.0
        score = min(base_score + type_bonus, 1.0)
        candidate_scores[node_id] = max(candidate_scores.get(node_id, 0.0), score)
        candidate_features[node_id].add(feature)
        if extra_feature is not None:
            candidate_features[node_id].add(extra_feature)
        if type_bonus > 0.0:
            candidate_features[node_id].add("type_bonus")

    def _rank_candidates(
        self,
        candidate_scores: dict[str, float],
        candidate_features: dict[str, set[str]],
    ) -> list[EntityGroundingCandidate]:
        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: (-item[1], self.catalog.records_by_id[item[0]].node_name, item[0]),
        )
        candidates: list[EntityGroundingCandidate] = []
        for node_id, score in ranked[: self.top_k]:
            record = self.catalog.records_by_id[node_id]
            candidates.append(
                EntityGroundingCandidate(
                    node_id=node_id,
                    node_name=record.node_name,
                    node_type=record.node_type,
                    source=record.source,
                    match_features=sorted(candidate_features[node_id]),
                    score=round(float(score), 4),
                )
            )
        return candidates


class EntityGroundingSelection(BaseModel):
    surface: str
    normalized_surface: str
    entity_type: str
    decision: Literal["link", "skip"]
    chosen_node_id: str = "none"
    chosen_node_name: str = "none"
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class EntityGroundingSelectionBatch(BaseModel):
    decisions: list[EntityGroundingSelection] = Field(default_factory=list)


class EntityGroundingJudge(JudgeBase):
    def __init__(self, model_name: str = "heuristic", device: str = "cpu") -> None:
        super().__init__(
            model_name=model_name,
            system_prompt=(
                "You validate PrimeKG entity grounding for medical QA seed building. "
                "Only choose a node from supplied candidates or skip. "
                "Never invent node IDs."
            ),
            device=device,
        )

    def evaluate(
        self,
        *,
        question_text: str,
        entities: Sequence[QuestionEntity],
        deterministic_decisions: Sequence[EntityGroundingDecision],
    ) -> EntityGroundingSelectionBatch:
        prompt = _build_entity_grounding_prompt(
            question_text=question_text,
            entities=entities,
            deterministic_decisions=deterministic_decisions,
        )
        fallback_payload = {
            "decisions": [
                {
                    "surface": decision.surface,
                    "normalized_surface": normalize_lookup_text(decision.surface) or normalize_basic_text(decision.surface),
                    "entity_type": decision.entity_type,
                    "decision": "link" if decision.linked_node_id else "skip",
                    "chosen_node_id": decision.linked_node_id or "none",
                    "chosen_node_name": decision.linked_node_name or "none",
                    "confidence": _decision_confidence(decision),
                    "rationale": decision.decision_reason,
                }
                for decision in deterministic_decisions
            ]
        }
        return self._call_llm(prompt=prompt, schema=EntityGroundingSelectionBatch, fallback_payload=fallback_payload)


class LLMAssistedSeedGrounder:
    def __init__(
        self,
        catalog: PrimeKGNodeCatalog,
        *,
        model_name: str = "heuristic",
        device: str = "cpu",
        top_k: int = DEFAULT_GROUNDING_TOP_K,
    ) -> None:
        self.catalog = catalog
        self.deterministic = DeterministicSeedGrounder(catalog, top_k=top_k)
        self.judge = EntityGroundingJudge(model_name=model_name, device=device)
        self.llm_used = self.judge.backend != "heuristic"
        self.backend_used = f"llm_assisted_v1_{self.judge.backend}" if self.llm_used else self.deterministic.backend_used
        self.fallback_reason: str | None = None if self.llm_used else "llm_backend_unavailable"

    def ground(self, question: str, entities: Sequence[QuestionEntity]) -> list[EntityGroundingDecision]:
        deterministic_decisions = self.deterministic.ground(question=question, entities=entities)
        if not self.llm_used:
            return deterministic_decisions

        result = self.judge.evaluate(
            question_text=question,
            entities=entities,
            deterministic_decisions=deterministic_decisions,
        )
        decision_lookup = {
            (normalize_basic_text(item.surface), item.entity_type): item
            for item in result.decisions
        }
        merged_decisions: list[EntityGroundingDecision] = []
        fallback_reasons: list[str] = []

        for deterministic in deterministic_decisions:
            key = (normalize_basic_text(deterministic.surface), deterministic.entity_type)
            selection = decision_lookup.get(key)
            allowed_candidates = {candidate.node_id: candidate for candidate in deterministic.candidates}
            if selection is None:
                merged_decisions.append(
                    _copy_decision(
                        deterministic,
                        backend_used=self.backend_used,
                        decision_reason="llm_missing_decision_fallback",
                    )
                )
                fallback_reasons.append("llm_missing_decision")
                continue

            chosen_node_id = selection.chosen_node_id if selection.chosen_node_id != "none" else None
            if selection.decision == "link" and chosen_node_id in allowed_candidates:
                chosen_candidate = allowed_candidates[chosen_node_id]
                merged_decisions.append(
                    EntityGroundingDecision(
                        surface=deterministic.surface,
                        entity_type=deterministic.entity_type,
                        cui=deterministic.cui,
                        linked_node_id=chosen_candidate.node_id,
                        linked_node_name=chosen_candidate.node_name,
                        backend_used=self.backend_used,
                        decision_reason="llm_link",
                        candidates=deterministic.candidates,
                    )
                )
                continue

            if selection.decision == "skip":
                merged_decisions.append(
                    EntityGroundingDecision(
                        surface=deterministic.surface,
                        entity_type=deterministic.entity_type,
                        cui=deterministic.cui,
                        linked_node_id=None,
                        linked_node_name=None,
                        backend_used=self.backend_used,
                        decision_reason="llm_skip",
                        candidates=deterministic.candidates,
                    )
                )
                continue

            fallback_reasons.append("llm_invalid_choice")
            if deterministic.linked_node_id is not None:
                merged_decisions.append(
                    _copy_decision(
                        deterministic,
                        backend_used=self.backend_used,
                        decision_reason="llm_invalid_choice_fallback",
                    )
                )
            else:
                merged_decisions.append(
                    EntityGroundingDecision(
                        surface=deterministic.surface,
                        entity_type=deterministic.entity_type,
                        cui=deterministic.cui,
                        linked_node_id=None,
                        linked_node_name=None,
                        backend_used=self.backend_used,
                        decision_reason="llm_invalid_choice_unresolved",
                        candidates=deterministic.candidates,
                    )
                )
        self.fallback_reason = ", ".join(sorted(set(fallback_reasons))) if fallback_reasons else None
        return merged_decisions


def serialize_grounding_decisions(decisions: Sequence[EntityGroundingDecision]) -> list[dict[str, Any]]:
    return [
        {
            "surface": decision.surface,
            "entity_type": decision.entity_type,
            "cui": decision.cui,
            "linked_node_id": decision.linked_node_id,
            "linked_node_name": decision.linked_node_name,
            "backend_used": decision.backend_used,
            "decision_reason": decision.decision_reason,
            "candidates": [
                {
                    "node_id": candidate.node_id,
                    "node_name": candidate.node_name,
                    "node_type": candidate.node_type,
                    "source": candidate.source,
                    "match_features": list(candidate.match_features),
                    "score": float(candidate.score),
                }
                for candidate in decision.candidates
            ],
        }
        for decision in decisions
    ]


def summarize_grounding_candidates(decisions: Sequence[EntityGroundingDecision]) -> dict[str, Any]:
    candidate_counts = [len(decision.candidates) for decision in decisions]
    linked_count = sum(1 for decision in decisions if decision.linked_node_id is not None)
    fuzzy_link_count = sum(
        1
        for decision in decisions
        if decision.linked_node_id is not None
        and any("fuzzy_similarity" in candidate.match_features for candidate in decision.candidates[:1])
    )
    return {
        "entity_count": len(decisions),
        "linked_count": linked_count,
        "unresolved_count": len(decisions) - linked_count,
        "avg_candidates": round(sum(candidate_counts) / len(candidate_counts), 4) if candidate_counts else 0.0,
        "max_candidates": max(candidate_counts, default=0),
        "fuzzy_link_count": fuzzy_link_count,
    }


def _decision_confidence(decision: EntityGroundingDecision) -> float:
    if not decision.candidates:
        return 0.0
    return float(decision.candidates[0].score)


def _entity_matches_node_type(entity_type: str, node_type: str) -> bool:
    normalized_node_type = NODE_TYPE_TO_ENTITY_TYPE.get(normalize_basic_text(node_type), "other")
    return entity_type == normalized_node_type or entity_type == "other"


def _token_overlap_count(query_tokens: Sequence[str], node_tokens: Sequence[str]) -> int:
    return len(set(query_tokens) & set(node_tokens))


def _token_containment_score(query_lookup: str, node_lookup: str) -> float:
    query_tokens = [token for token in query_lookup.split() if len(token) >= DEFAULT_MIN_TOKEN_LENGTH]
    node_tokens = set(token for token in node_lookup.split() if len(token) >= DEFAULT_MIN_TOKEN_LENGTH)
    if query_tokens and all(token in node_tokens for token in query_tokens):
        return DEFAULT_TOKEN_CONTAINMENT_SCORE
    return 0.0


def _fuzzy_score(query_lookup: str, node_lookup: str) -> float:
    if not query_lookup or not node_lookup:
        return 0.0
    sequence_ratio = SequenceMatcher(a=query_lookup, b=node_lookup).ratio()
    trigram_ratio = _char_trigram_cosine(query_lookup, node_lookup)
    return 0.5 * sequence_ratio + 0.5 * trigram_ratio


def _char_trigram_cosine(left: str, right: str) -> float:
    left_counts = Counter(_char_trigrams(left))
    right_counts = Counter(_char_trigrams(right))
    if not left_counts or not right_counts:
        return 0.0
    numerator = sum(left_counts[token] * right_counts[token] for token in left_counts.keys() & right_counts.keys())
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    denominator = max(left_norm * right_norm, 1e-8)
    return float(numerator / denominator)


def _char_trigrams(value: str) -> list[str]:
    padded = f"  {value}  "
    return [padded[index : index + 3] for index in range(max(len(padded) - 2, 0))]


def _copy_decision(
    decision: EntityGroundingDecision,
    *,
    backend_used: str,
    decision_reason: str,
) -> EntityGroundingDecision:
    return EntityGroundingDecision(
        surface=decision.surface,
        entity_type=decision.entity_type,
        cui=decision.cui,
        linked_node_id=decision.linked_node_id,
        linked_node_name=decision.linked_node_name,
        backend_used=backend_used,
        decision_reason=decision_reason,
        candidates=list(decision.candidates),
    )


def _build_entity_grounding_prompt(
    *,
    question_text: str,
    entities: Sequence[QuestionEntity],
    deterministic_decisions: Sequence[EntityGroundingDecision],
) -> str:
    candidate_payload = [
        {
            "surface": entity.surface,
            "entity_type": entity.entity_type,
            "cui": entity.cui,
            "deterministic_decision": decision.decision_reason,
            "deterministic_linked_node_id": decision.linked_node_id or "none",
            "deterministic_linked_node_name": decision.linked_node_name or "none",
            "candidates": [
                {
                    "node_id": candidate.node_id,
                    "node_name": candidate.node_name,
                    "node_type": candidate.node_type,
                    "source": candidate.source,
                    "score": candidate.score,
                    "match_features": candidate.match_features,
                }
                for candidate in decision.candidates
            ],
        }
        for entity, decision in zip(entities, deterministic_decisions)
    ]
    return (
        f"Question:\n{question_text}\n\n"
        f"Extracted entities and candidate PrimeKG nodes:\n{candidate_payload}\n\n"
        "For each entity, either choose one supplied candidate node_id or return skip. "
        "Never invent node IDs, aliases, or node names."
    )


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    return None


def _clean_seed_surface(value: str) -> str:
    cleaned = STRUCTURAL_SEGMENT_PATTERN.sub(" ", value or "")
    cleaned = re.sub(r"^\s*[A-H][\.\)]\s*", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t\n\r:;,-")
    return cleaned


def _is_seed_surface_candidate(value: str) -> bool:
    cleaned = _clean_seed_surface(value)
    if not cleaned:
        return False
    normalized = normalize_lookup_text(cleaned)
    if not normalized:
        return False
    tokens = normalized.split()
    if not tokens:
        return False
    if len(tokens) == 1 and (tokens[0] in SEED_EXTRACTION_STOPWORDS or len(tokens[0]) < DEFAULT_MIN_TOKEN_LENGTH):
        return False
    non_stop_tokens = [token for token in tokens if token not in SEED_EXTRACTION_STOPWORDS]
    return bool(non_stop_tokens)


def _dedupe_preserve_order(values: Sequence[T]) -> list[T]:
    deduped: list[T] = []
    seen: set[T] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _query_variants(entity: QuestionEntity) -> list[tuple[str, str, str]]:
    variants: list[tuple[str, str, str]] = []
    seen_lookup_keys: set[str] = set()

    def _append_variant(surface: str, feature: str) -> None:
        basic = normalize_basic_text(surface)
        lookup = normalize_lookup_text(surface)
        if not lookup or lookup in seen_lookup_keys:
            return
        seen_lookup_keys.add(lookup)
        variants.append((basic, lookup, feature))

    _append_variant(entity.surface, "surface_form")
    if entity.entity_type == "anatomy":
        normalized_lookup = normalize_lookup_text(entity.surface)
        anatomy_rewrite = _rewrite_anatomy_lookup(normalized_lookup)
        if anatomy_rewrite and anatomy_rewrite != normalized_lookup:
            _append_variant(anatomy_rewrite, "anatomy_token_rewrite")
        for alias_surface in ANATOMY_QUERY_ALIASES.get(normalized_lookup, ()):
            _append_variant(alias_surface, "anatomy_phrase_alias")
    return variants


def _rewrite_anatomy_lookup(value: str) -> str:
    if not value:
        return value
    rewritten_tokens = [ANATOMY_QUERY_TOKEN_ALIASES.get(token, token) for token in value.split()]
    rewritten = " ".join(rewritten_tokens).strip()
    if rewritten.startswith("sphincter ") and len(rewritten_tokens) == 2:
        rewritten = f"{rewritten_tokens[1]} {rewritten_tokens[0]}"
    return rewritten


__all__ = [
    "DEFAULT_GEMINI_GROUNDER",
    "DEFAULT_GROUNDING_TOP_K",
    "DEFAULT_OPENAI_GROUNDER",
    "DeterministicSeedGrounder",
    "EntityGroundingCandidate",
    "EntityGroundingDecision",
    "EntityGroundingJudge",
    "EntityGroundingSelection",
    "EntityGroundingSelectionBatch",
    "GROUNDING_MODES",
    "LightweightSeedEntityExtractor",
    "LLMAssistedSeedGrounder",
    "PrimeKGNodeCatalog",
    "PrimeKGNodeRecord",
    "SEED_EXTRACTOR_BACKEND",
    "SeedEntityExtractor",
    "SeedGrounder",
    "serialize_grounding_decisions",
    "summarize_grounding_candidates",
]
