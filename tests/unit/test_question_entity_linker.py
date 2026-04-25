"""Unit tests for the Phase 4 QuestionEntityLinker and its components."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from minimed_rag.kg.graph_client import Node
from minimed_rag.nlp.linking.entity_linker import (
    LinkedMention,
    QuestionEntityLinker,
)
from minimed_rag.nlp.linking.scispacy_entity_linker import ScispacyEntityLinker
from minimed_rag.nlp.linking.synonym_fallback_linker import (
    SynonymFallbackLinker,
    normalize_text,
)
from minimed_rag.storage.repositories.term_repo import TermMatch


def _ent(text: str, label: str, kb_ents: list[tuple[str, float]] | None = None):
    return SimpleNamespace(
        text=text,
        label_=label,
        _=SimpleNamespace(kb_ents=kb_ents or []),
    )


class _FakeNlp:
    def __init__(self, ents_per_text: dict[str, list]):
        self.ents_per_text = ents_per_text
        self.pipe_names: list[str] = []

    def __call__(self, text: str):
        return SimpleNamespace(ents=self.ents_per_text.get(text, []))


def test_scispacy_linker_attaches_cui_when_above_threshold():
    nlp = _FakeNlp({"metformin": [_ent("metformin", "CHEMICAL", [("C0025598", 0.95)])]})
    linker = ScispacyEntityLinker(nlp_model=nlp, use_umls_linker=True, threshold=0.7)
    mentions = linker.link("metformin")

    assert len(mentions) == 1
    m = mentions[0]
    assert m.cui == "C0025598"
    assert m.concept_id == "UMLS:C0025598"
    assert m.confidence == 0.95
    assert m.semantic_type == "CHEMICAL"


def test_scispacy_linker_leaves_cui_none_when_below_threshold():
    nlp = _FakeNlp({"metformin": [_ent("metformin", "CHEMICAL", [("C0025598", 0.3)])]})
    linker = ScispacyEntityLinker(nlp_model=nlp, threshold=0.7)
    m = linker.link("metformin")[0]
    assert m.cui is None
    assert m.semantic_type == "CHEMICAL"
    assert m.confidence == 0.0


def test_scispacy_linker_no_entities_returns_empty():
    nlp = _FakeNlp({"hi": []})
    linker = ScispacyEntityLinker(nlp_model=nlp)
    assert linker.link("hi") == []


@dataclass
class FakeTermRepo:
    rows: dict[str, list[TermMatch]]

    def search_by_normalized_text(self, normalized: str, limit: int = 10):
        return self.rows.get(normalized, [])[:limit]


def test_synonym_fallback_returns_first_match():
    repo = FakeTermRepo(
        rows={
            "metformin": [
                TermMatch(
                    term_id="UMLS:AUI:A1",
                    concept_id="UMLS:C0025598",
                    cui="C0025598",
                    text="Metformin",
                    normalized_text="metformin",
                    language="ENG",
                    source_system="DRUGBANK",
                    preferred_name="Metformin",
                )
            ]
        }
    )
    linker = SynonymFallbackLinker(term_repo=repo, confidence=0.7)
    m = linker.link_mention("Metformin")
    assert m is not None
    assert m.cui == "C0025598"
    assert m.concept_id == "UMLS:C0025598"
    assert m.confidence == 0.7


def test_synonym_fallback_returns_none_when_no_match():
    repo = FakeTermRepo(rows={})
    assert SynonymFallbackLinker(term_repo=repo).link_mention("unknown term") is None


def test_normalize_text_collapses_whitespace_and_lowers():
    assert normalize_text("  Type 2   DIABETES  ") == "type 2 diabetes"


@dataclass
class FakeScispacy:
    mentions: list[LinkedMention]

    def link(self, text: str):
        return list(self.mentions)


@dataclass
class FakeFallback:
    map: dict[str, LinkedMention]

    def link_mention(self, mention_text: str):
        return self.map.get(mention_text.lower())


@dataclass
class FakeGraphClient:
    by_cui: dict[str, list[Node]]

    def find_entity_by_cui(self, cui: str, limit: int = 1):
        return self.by_cui.get(cui, [])[:limit]


def test_question_linker_prefers_scispacy_when_confident():
    scispacy = FakeScispacy(
        mentions=[
            LinkedMention(
                mention_text="metformin",
                expanded_text="metformin",
                entity_id="",
                concept_id="UMLS:C0025598",
                cui="C0025598",
                semantic_type="CHEMICAL",
                confidence=0.95,
            )
        ]
    )
    graph = FakeGraphClient(by_cui={"C0025598": [Node(labels=("Entity",), entity_id="KG:Drug:m")]})
    linker = QuestionEntityLinker(
        scispacy_linker=scispacy,
        fallback_linker=FakeFallback(map={}),
        graph_client=graph,
        min_confidence=0.5,
    )
    mentions = linker.link_question("What is metformin used for?")
    assert len(mentions) == 1
    m = mentions[0]
    assert m.cui == "C0025598"
    assert m.entity_id == "KG:Drug:m"


def test_question_linker_falls_back_to_postgres_when_scispacy_low_confidence():
    scispacy = FakeScispacy(
        mentions=[
            LinkedMention(
                mention_text="Metformin",
                expanded_text="Metformin",
                entity_id="",
                concept_id=None,
                cui=None,
                semantic_type="CHEMICAL",
                confidence=0.0,
            )
        ]
    )
    fallback = FakeFallback(
        map={
            "metformin": LinkedMention(
                mention_text="Metformin",
                expanded_text="Metformin",
                entity_id="",
                concept_id="UMLS:C0025598",
                cui="C0025598",
                semantic_type=None,
                confidence=0.7,
            )
        }
    )
    graph = FakeGraphClient(by_cui={})
    linker = QuestionEntityLinker(
        scispacy_linker=scispacy,
        fallback_linker=fallback,
        graph_client=graph,
        min_confidence=0.5,
    )
    mentions = linker.link_question("Metformin question")
    assert len(mentions) == 1
    # Fallback match inherits semantic_type from scispacy.
    assert mentions[0].cui == "C0025598"
    assert mentions[0].semantic_type == "CHEMICAL"


def test_question_linker_keeps_ner_only_mentions_when_no_cui():
    scispacy = FakeScispacy(
        mentions=[
            LinkedMention(
                mention_text="AMPK",
                expanded_text="AMPK",
                entity_id="",
                concept_id=None,
                cui=None,
                semantic_type="GENE_OR_GENE_PRODUCT",
                confidence=0.0,
            )
        ]
    )
    linker = QuestionEntityLinker(
        scispacy_linker=scispacy,
        fallback_linker=FakeFallback(map={}),
        graph_client=FakeGraphClient(by_cui={}),
    )
    mentions = linker.link_question("AMPK signaling")
    assert len(mentions) == 1
    assert mentions[0].cui is None
    assert mentions[0].semantic_type == "GENE_OR_GENE_PRODUCT"


def test_question_linker_drops_unresolvable_mentions_without_semantic_type():
    scispacy = FakeScispacy(
        mentions=[
            LinkedMention(
                mention_text="blah",
                expanded_text="blah",
                entity_id="",
                concept_id=None,
                cui=None,
                semantic_type=None,
                confidence=0.0,
            )
        ]
    )
    linker = QuestionEntityLinker(
        scispacy_linker=scispacy,
        fallback_linker=FakeFallback(map={}),
        graph_client=FakeGraphClient(by_cui={}),
    )
    assert linker.link_question("blah") == []
