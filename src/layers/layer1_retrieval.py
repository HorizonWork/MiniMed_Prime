from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence, get_args

from src.schemas import EvidenceBundle, KGEdge, PrimeKGRelation, PubMedPassage, QuestionEntity, QuestionType
from src.utils.kaggle_env import KaggleEnv
from src.utils.pubmed_client import PubMedArticle, PubMedClient
from src.utils.structured_logger import DEFAULT_LOG_DIR, StructuredLogger

import networkx as nx
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

DEFAULT_PRIMEKG_PATH = KaggleEnv.path("data/kg/primekg")
DEFAULT_PUBMED_CACHE_PATH = KaggleEnv.ensure_writeable(KaggleEnv.path("data/pubmed_cache.jsonl"))
DEFAULT_SCISPACY_MODEL = "en_core_sci_lg"
DEFAULT_ENTITY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MEDCPT_QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
DEFAULT_MEDCPT_ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"
DEFAULT_MEDCPT_CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"
DEFAULT_PUBMED_TOP_K = 5
DEFAULT_PUBMED_SEARCH_CANDIDATES = 50
DEFAULT_PRIMEKG_TOP_K = 200
DEFAULT_PPR_ALPHA = 0.85
DEFAULT_EDGE_SEED_BONUS = 0.05
DEFAULT_EDGE_SOURCE_WEIGHT = 0.10
DEFAULT_EDGE_NODE_WEIGHT = 0.45
DEFAULT_BM25_WEIGHT = 0.4
DEFAULT_DENSE_WEIGHT = 0.3
DEFAULT_CROSS_ENCODER_WEIGHT = 0.3
DEFAULT_CROSS_ENCODER_POOL = 50
DEFAULT_ENTITY_CONTEXT_WEIGHT = 0.25
DEFAULT_ENTITY_LINKER_SCORE_WEIGHT = 0.75
DEFAULT_SOURCE_RELIABILITY = 0.70
DEFAULT_MAX_ENTITY_NGRAM = 3
DEFAULT_MIN_ENTITY_TOKEN_LENGTH = 3
PRIMEKG_FILE_PRIORITY = (
    "kg.csv",
    "kg_raw.csv",
    "kg_grouped.csv",
    "kg_giant.csv",
    "primekg.csv",
)

QUESTION_TYPE_PATTERNS: dict[QuestionType, tuple[str, ...]] = {
    "drug_interaction": ("interaction", "interact", "contraindicated", "combine", "co-administer", "coadminister"),
    "dosage": ("dose", "dosage", "mg", "mcg", "titrate", "how much"),
    "diagnosis": ("diagnosis", "diagnose", "differential", "most likely", "presents with", "likely condition"),
    "etiology": ("cause", "causes", "caused by", "etiology", "why does", "mechanism"),
    "factoid": (),
    "other": (),
}

QUESTION_TYPE_RELATION_FILTERS: dict[QuestionType, set[str]] = {
    "drug_interaction": {"drug_drug", "contraindication", "drug_effect", "drug_protein"},
    "etiology": {
        "disease_disease",
        "disease_protein",
        "exposure_disease",
        "exposure_protein",
        "phenotype_protein",
    },
    "diagnosis": {
        "disease_phenotype_negative",
        "disease_phenotype_positive",
        "phenotype_phenotype",
        "disease_disease",
    },
    "dosage": {"drug_protein", "drug_effect", "contraindication"},
    "factoid": set(),
    "other": set(),
}

NODE_TYPE_TO_ENTITY_TYPE = {
    "disease": "disease",
    "drug": "drug",
    "protein": "protein",
    "phenotype": "symptom",
    "symptom": "symptom",
    "anatomy": "anatomy",
}

SOURCE_RELIABILITY_LOOKUP = {
    "drugbank": 0.95,
    "drugcentral": 0.94,
    "guidetopharmacology": 0.93,
    "omim": 0.90,
    "disgenet": 0.88,
    "reactome": 0.86,
    "go": 0.84,
    "hpo": 0.83,
    "ctd": 0.80,
    "primekg": 0.78,
    "pubmed": 0.75,
}

ENTITY_TYPE_LEXICON = {
    "drug": {
        "aspirin": "C0004057",
        "warfarin": "C0043031",
        "ibuprofen": "C0020740",
        "acetaminophen": "C0000970",
        "metformin": "C0025598",
        "insulin": "C0021641",
        "heparin": "C0019079",
        "amoxicillin": "C0002639",
        "lisinopril": "C0070222",
        "atorvastatin": "C0286651",
        "clopidogrel": "C0070166",
        "apixaban": "C2004428",
        "rivaroxaban": "C2699778",
    },
    "disease": {
        "stroke": "C0038454",
        "atrial fibrillation": "C0004238",
        "diabetes": "C0011849",
        "hypertension": "C0020538",
        "myocardial infarction": "C0027051",
        "pneumonia": "C0032285",
        "asthma": "C0004096",
        "heart failure": "C0018801",
        "covid-19": "C5203670",
    },
    "symptom": {
        "fever": "C0015967",
        "cough": "C0010200",
        "rash": "C0037284",
        "chest pain": "C0008031",
        "headache": "C0018681",
        "nausea": "C0027497",
    },
    "anatomy": {
        "liver": "C0023884",
        "kidney": "C0022646",
        "heart": "C0018787",
        "lung": "C0024109",
        "brain": "C0006104",
    },
    "protein": {
        "egfr": "C3812682",
        "braf": "C3542021",
        "ace2": "C5193402",
        "tnf-alpha": "C0242493",
        "il-6": "C1716360",
    },
}

HEURISTIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "does",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "of",
    "on",
    "or",
    "patient",
    "patients",
    "the",
    "to",
    "what",
    "when",
    "with",
}


@dataclass(slots=True)
class LexiconEntry:
    surface: str
    cui: str | None
    entity_type: str


@dataclass(slots=True)
class EntityLinkerConfig:
    scispacy_model: str = DEFAULT_SCISPACY_MODEL
    embedding_model_name: str | None = DEFAULT_ENTITY_EMBEDDING_MODEL
    max_ngram: int = DEFAULT_MAX_ENTITY_NGRAM
    min_token_length: int = DEFAULT_MIN_ENTITY_TOKEN_LENGTH
    linker_score_weight: float = DEFAULT_ENTITY_LINKER_SCORE_WEIGHT
    context_similarity_weight: float = DEFAULT_ENTITY_CONTEXT_WEIGHT


@dataclass(slots=True)
class PrimeKGExtractorConfig:
    primekg_path: Path
    ppr_alpha: float = DEFAULT_PPR_ALPHA
    top_k: int = DEFAULT_PRIMEKG_TOP_K
    source_reliability_lookup: dict[str, float] = field(default_factory=lambda: dict(SOURCE_RELIABILITY_LOOKUP))
    default_source_reliability: float = DEFAULT_SOURCE_RELIABILITY
    edge_seed_bonus: float = DEFAULT_EDGE_SEED_BONUS
    edge_source_weight: float = DEFAULT_EDGE_SOURCE_WEIGHT
    edge_node_weight: float = DEFAULT_EDGE_NODE_WEIGHT


@dataclass(slots=True)
class PubMedRetrieverConfig:
    api_key: str | None = None
    cache_path: Path = field(default_factory=lambda: DEFAULT_PUBMED_CACHE_PATH)
    search_candidate_count: int = DEFAULT_PUBMED_SEARCH_CANDIDATES
    top_k: int = DEFAULT_PUBMED_TOP_K
    bm25_weight: float = DEFAULT_BM25_WEIGHT
    dense_weight: float = DEFAULT_DENSE_WEIGHT
    cross_encoder_weight: float = DEFAULT_CROSS_ENCODER_WEIGHT
    cross_encoder_pool: int = DEFAULT_CROSS_ENCODER_POOL
    query_encoder_model_name: str | None = DEFAULT_MEDCPT_QUERY_ENCODER
    article_encoder_model_name: str | None = DEFAULT_MEDCPT_ARTICLE_ENCODER
    cross_encoder_model_name: str | None = DEFAULT_MEDCPT_CROSS_ENCODER


@dataclass(slots=True)
class AgenticRetrieverConfig:
    primekg_path: Path = field(default_factory=lambda: DEFAULT_PRIMEKG_PATH)
    pubmed_cache_path: Path = field(default_factory=lambda: DEFAULT_PUBMED_CACHE_PATH)
    scispacy_model: str = DEFAULT_SCISPACY_MODEL
    primekg_top_k: int = DEFAULT_PRIMEKG_TOP_K
    pubmed_top_k: int = DEFAULT_PUBMED_TOP_K
    relation_filter: set[str] | None = None
    pubmed_api_key: str | None = None


class EntityLinker:
    """Entity linker with optional scispaCy/UMLS support and a rule-based fallback."""

    def __init__(self, scispacy_model: str = DEFAULT_SCISPACY_MODEL) -> None:
        self.config = EntityLinkerConfig(scispacy_model=scispacy_model)
        self._lexicon_entries = self._build_lexicon_entries()
        self._lexicon_lookup = {entry.surface: entry for entry in self._lexicon_entries}
        self._embedding_model = None
        self._umls_linker = None
        self._uses_umls = False
        self.event_logger = StructuredLogger("layer1_retrieval", DEFAULT_LOG_DIR)
        self._nlp = self._build_nlp()

    @property
    def backend_used(self) -> str:
        return "scispacy_umls" if self._uses_umls else "rule_based"

    def link(self, text: str) -> list[QuestionEntity]:
        """Link question spans to coarse medical entities."""

        started_at = time.perf_counter()
        text = text.strip()
        if not text:
            return []

        entities = self._link_with_umls(text) if self._uses_umls else []
        if not entities:
            entities = self._link_with_rules(text)

        deduplicated_entities: list[QuestionEntity] = []
        seen_keys: set[tuple[str, str | None, str]] = set()
        for entity in entities:
            key = (entity.surface.lower(), entity.cui, entity.entity_type)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduplicated_entities.append(entity)
        self.event_logger.log_event(
            "entity_linking",
            {
                "entities_found": len(deduplicated_entities),
                "scispacy_used": self._uses_umls,
                "fallback_triggered": not self._uses_umls,
                "backend_used": self.backend_used,
                "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            },
        )
        return deduplicated_entities

    def _build_nlp(self) -> Any:
        try:
            import spacy
            from spacy.language import Language
        except ImportError:
            logger.warning("spaCy is unavailable; EntityLinker will use regex-only fallback.")
            self.event_logger.log_event(
                "entity_linking_backend_fallback",
                {
                    "reason": "spacy_unavailable",
                    "backend_used": "rule_based",
                    "latency_ms": 0.0,
                },
            )
            return None

        try:
            import scispacy  # noqa: F401
            from scispacy.linking import EntityLinker as ScispaCyEntityLinker

            nlp = spacy.load(self.config.scispacy_model)
            if "scispacy_linker" not in nlp.pipe_names:
                nlp.add_pipe(
                    "scispacy_linker",
                    config={"resolve_abbreviations": True, "linker_name": "umls"},
                )
            self._umls_linker = nlp.get_pipe("scispacy_linker")
            self._uses_umls = True
            return nlp
        except Exception as exc:
            fallback_reason, expected_fallback = self._classify_scispacy_exception(exc)
            payload = {
                "reason": fallback_reason,
                "backend_used": "rule_based",
                "latency_ms": 0.0,
                "expected_fallback": expected_fallback,
                "scispacy_model": self.config.scispacy_model,
            }
            if expected_fallback:
                logger.info(
                    "EntityLinker is using rule-based fallback because scispaCy/UMLS is unavailable: %s",
                    exc,
                )
                self.event_logger.log_event(
                    "entity_linking_backend_fallback",
                    {
                        **payload,
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                )
            else:
                logger.warning("Falling back to rule-based entity linking because scispaCy initialization failed: %s", exc)
                self.event_logger.log_exception("entity_linking_backend_fallback", exc, payload)

        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler")
            ruler.add_patterns(
                [{"label": entry.entity_type.upper(), "pattern": entry.surface} for entry in self._lexicon_entries]
            )
        return nlp

    def _classify_scispacy_exception(self, exc: Exception) -> tuple[str, bool]:
        message = str(exc).lower()
        module_name = (getattr(exc, "name", None) or "").lower()
        if isinstance(exc, ModuleNotFoundError) and (
            module_name == "scispacy" or "no module named 'scispacy'" in message
        ):
            return "scispacy_package_missing", True
        if isinstance(exc, OSError) and "can't find model" in message and self.config.scispacy_model.lower() in message:
            return "scispacy_model_missing", True
        return "scispacy_initialization_failed", False

    def _build_lexicon_entries(self) -> list[LexiconEntry]:
        entries: list[LexiconEntry] = []
        for entity_type, term_mapping in ENTITY_TYPE_LEXICON.items():
            for surface, cui in term_mapping.items():
                entries.append(LexiconEntry(surface=surface, cui=cui, entity_type=entity_type))
        return entries

    def _link_with_umls(self, text: str) -> list[QuestionEntity]:
        if self._nlp is None or self._umls_linker is None:
            return []

        doc = self._nlp(text)
        entities: list[QuestionEntity] = []
        for span in doc.ents:
            candidates = getattr(span._, "kb_ents", ())
            best_candidate = self._select_best_umls_candidate(candidates=candidates, question_text=text)
            entity_type = self._infer_entity_type(span.text)
            cui = best_candidate[0] if best_candidate is not None else None
            entities.append(
                QuestionEntity(
                    surface=span.text,
                    cui=cui,
                    primekg_node_id=None,
                    entity_type=entity_type,
                )
            )
        return entities

    def _select_best_umls_candidate(
        self,
        candidates: Sequence[tuple[str, float]],
        question_text: str,
    ) -> tuple[str, float] | None:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        candidate_texts: list[str] = []
        candidate_scores: list[float] = []
        for cui, score in candidates:
            kb_entity = self._umls_linker.kb.cui_to_entity.get(cui) if self._umls_linker is not None else None
            candidate_text = cui
            if kb_entity is not None:
                definition = getattr(kb_entity, "definition", "") or ""
                candidate_text = " ".join(part for part in [kb_entity.canonical_name, definition] if part).strip() or cui
            candidate_texts.append(candidate_text)
            candidate_scores.append(float(score))

        embedding_scores = self._cosine_similarity_against_context(question_text, candidate_texts)
        normalized_candidate_scores = self._normalize_scores(candidate_scores)
        combined_scores = (
            (self.config.linker_score_weight * normalized_candidate_scores)
            + (self.config.context_similarity_weight * embedding_scores)
        )
        best_index = int(np.argmax(combined_scores))
        return candidates[best_index]

    def _link_with_rules(self, text: str) -> list[QuestionEntity]:
        entities: list[QuestionEntity] = []
        normalized_text = self._normalize_surface(text)

        if self._nlp is not None:
            doc = self._nlp(text)
            for span in doc.ents:
                normalized_surface = self._normalize_surface(span.text)
                lexicon_entry = self._lexicon_lookup.get(normalized_surface)
                entity_type = lexicon_entry.entity_type if lexicon_entry is not None else self._infer_entity_type(span.text)
                cui = lexicon_entry.cui if lexicon_entry is not None else None
                entities.append(
                    QuestionEntity(
                        surface=span.text,
                        cui=cui,
                        primekg_node_id=None,
                        entity_type=entity_type,
                    )
                )

        seen_surfaces = {self._normalize_surface(entity.surface) for entity in entities}
        for phrase in self._iter_candidate_phrases(normalized_text):
            if phrase in seen_surfaces:
                continue
            lexicon_entry = self._lexicon_lookup.get(phrase)
            if lexicon_entry is not None:
                entities.append(
                    QuestionEntity(
                        surface=phrase,
                        cui=lexicon_entry.cui,
                        primekg_node_id=None,
                        entity_type=lexicon_entry.entity_type,  # type: ignore[arg-type]
                    )
                )
                continue

            if not self._looks_medical(phrase):
                continue
            entities.append(
                QuestionEntity(
                    surface=phrase,
                    cui=None,
                    primekg_node_id=None,
                    entity_type=self._infer_entity_type(phrase),
                )
            )
        return entities

    def _iter_candidate_phrases(self, text: str) -> Iterable[str]:
        tokens = re.findall(r"[A-Za-z0-9-]+", text.lower())
        for ngram_size in range(self.config.max_ngram, 0, -1):
            for start_index in range(0, len(tokens) - ngram_size + 1):
                ngram_tokens = tokens[start_index : start_index + ngram_size]
                if any(token in HEURISTIC_STOPWORDS for token in ngram_tokens):
                    continue
                phrase = " ".join(ngram_tokens)
                if len(phrase) < self.config.min_token_length:
                    continue
                yield phrase

    def _looks_medical(self, phrase: str) -> bool:
        if phrase in self._lexicon_lookup:
            return True
        lowered = phrase.lower()
        return any(
            lowered.endswith(suffix)
            for suffix in ("mab", "nib", "azole", "cycline", "mycin", "pril", "sartan", "olol", "statin")
        ) or lowered.isupper()

    def _infer_entity_type(self, surface: str) -> str:
        lowered = self._normalize_surface(surface)
        for entity_type, term_mapping in ENTITY_TYPE_LEXICON.items():
            if lowered in term_mapping:
                return entity_type
        if lowered.endswith(("mab", "nib", "statin", "pril", "sartan", "olol")):
            return "drug"
        if lowered in {"pain", "rash", "fever", "cough"}:
            return "symptom"
        if lowered in {"heart", "lung", "liver", "kidney", "brain"}:
            return "anatomy"
        if lowered.isupper() and 2 <= len(lowered) <= 8:
            return "protein"
        return "other"

    def _cosine_similarity_against_context(self, question_text: str, candidate_texts: Sequence[str]) -> np.ndarray:
        if not candidate_texts:
            return np.zeros(0, dtype=np.float32)
        embeddings = self._encode_texts([question_text, *candidate_texts])
        if embeddings is None:
            return np.zeros(len(candidate_texts), dtype=np.float32)
        context_embedding = embeddings[0]
        candidate_embeddings = embeddings[1:]
        context_norm = np.linalg.norm(context_embedding)
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
        denominator = np.maximum(context_norm * candidate_norms, 1e-8)
        similarities = np.dot(candidate_embeddings, context_embedding) / denominator
        return self._normalize_scores(similarities)

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray | None:
        if self.config.embedding_model_name is None:
            return None
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(self.config.embedding_model_name)
            except Exception as exc:
                logger.warning("SentenceTransformer model load failed for entity disambiguation: %s", exc)
                self.event_logger.log_exception(
                    "entity_linking_context_encoder_fallback",
                    exc,
                    {"backend_used": self.backend_used, "latency_ms": 0.0},
                )
                self.config.embedding_model_name = None
                return None
        embeddings = self._embedding_model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False)
        return np.asarray(embeddings, dtype=np.float32)

    @staticmethod
    def _normalize_surface(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    @staticmethod
    def _normalize_scores(scores: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(scores, dtype=np.float32)
        if array.size == 0:
            return array
        min_value = float(array.min())
        max_value = float(array.max())
        if math.isclose(min_value, max_value):
            return np.ones_like(array, dtype=np.float32) if not math.isclose(max_value, 0.0) else np.zeros_like(array)
        return (array - min_value) / (max_value - min_value)


class PrimeKGExtractor:
    """PrimeKG loader and ranked 2-hop subgraph extractor."""

    def __init__(self, primekg_path: Path) -> None:
        resolved_primekg_path = primekg_path if primekg_path.is_absolute() else KaggleEnv.path(primekg_path)
        self.config = PrimeKGExtractorConfig(primekg_path=resolved_primekg_path)
        self.graph = nx.MultiDiGraph()
        self.node_degrees: dict[str, int] = {}
        self.relation_type_mapping: dict[str, int] = {}
        self.edge_lookup: dict[str, KGEdge] = {}
        self.node_name_index: dict[str, set[str]] = defaultdict(set)
        self.node_cui_index: dict[str, set[str]] = defaultdict(set)
        self.event_logger = StructuredLogger("layer1_retrieval", DEFAULT_LOG_DIR)
        self.backend_used = "empty_graph"
        self._load_primekg()

    def resolve_seed_entities(self, seed_entities: Sequence[QuestionEntity]) -> list[QuestionEntity]:
        """Attach PrimeKG node ids to entities when a graph match is available."""

        if not self.graph:
            return [entity.model_copy(deep=True) for entity in seed_entities]

        resolved_entities: list[QuestionEntity] = []
        for entity in seed_entities:
            if entity.primekg_node_id and self.graph.has_node(entity.primekg_node_id):
                resolved_entities.append(entity.model_copy(deep=True))
                continue

            candidate_ids: set[str] = set()
            if entity.cui:
                candidate_ids.update(self.node_cui_index.get(entity.cui.lower(), set()))
            candidate_ids.update(self.node_name_index.get(self._normalize_text(entity.surface), set()))

            selected_node_id = self._select_best_node(candidate_ids=candidate_ids, entity=entity)
            resolved_entities.append(entity.model_copy(update={"primekg_node_id": selected_node_id}))
        return resolved_entities

    def extract_2hop(
        self,
        seed_entities: list[QuestionEntity],
        top_k: int = DEFAULT_PRIMEKG_TOP_K,
        relation_filter: set[str] | None = None,
    ) -> list[KGEdge]:
        """Extract a ranked 2-hop neighborhood around seed entities."""

        started_at = time.perf_counter()
        if not self.graph:
            return []

        resolved_entities = self.resolve_seed_entities(seed_entities)
        seed_node_ids = {entity.primekg_node_id for entity in resolved_entities if entity.primekg_node_id}
        if not seed_node_ids:
            return []

        allowed_relations = {relation.lower() for relation in relation_filter} if relation_filter else None
        candidate_edge_ids: set[str] = set()
        candidate_node_ids: set[str] = set(seed_node_ids)
        frontier = set(seed_node_ids)
        visited = set(seed_node_ids)
        for _ in range(2):
            next_frontier: set[str] = set()
            for node_id in frontier:
                for edge in self._iter_incident_edges(node_id=node_id, allowed_relations=allowed_relations):
                    candidate_edge_ids.add(edge.edge_id)
                    candidate_node_ids.update((edge.head, edge.tail))
                    next_frontier.update((edge.head, edge.tail))
            frontier = next_frontier - visited
            visited.update(next_frontier)

        if not candidate_edge_ids:
            return []

        ppr_scores = self._compute_personalized_pagerank(node_ids=candidate_node_ids, seed_node_ids=seed_node_ids)
        scored_edges = []
        for edge_id in candidate_edge_ids:
            edge = self.edge_lookup[edge_id]
            score = self._score_edge(edge=edge, ppr_scores=ppr_scores, seed_node_ids=seed_node_ids)
            scored_edges.append((score, edge))

        scored_edges.sort(
            key=lambda item: (
                item[0],
                item[1].source_reliability,
                item[1].amg_confidence,
                item[1].edge_id,
            ),
            reverse=True,
        )
        top_edges = [edge for _, edge in scored_edges[:top_k]]
        candidate_nodes = {edge.head for edge in top_edges} | {edge.tail for edge in top_edges}
        self.event_logger.log_event(
            "subgraph_extracted",
            {
                "nodes": len(candidate_nodes),
                "edges": len(top_edges),
                "hops": 2,
                "ppr_topk": top_k,
                "backend_used": "networkx_csv",
                "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            },
        )
        return top_edges

    def _load_primekg(self) -> None:
        started_at = time.perf_counter()
        edge_node_tables = self._resolve_primekg_edge_node_tables(self.config.primekg_path)
        if edge_node_tables is not None and self._load_primekg_from_edge_node_tables(*edge_node_tables, started_at=started_at):
            return

        primekg_path = self._resolve_primekg_file(self.config.primekg_path)
        if primekg_path is None:
            logger.warning(
                "PrimeKG path %s was not found. Graph extraction will return empty subgraphs.",
                self.config.primekg_path,
            )
            self.event_logger.log_event(
                "primekg_missing",
                {
                    "primekg_path": str(self.config.primekg_path),
                    "backend_used": "empty_graph",
                    "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                },
            )
            return

        frame = pd.read_csv(primekg_path)
        if frame.empty:
            logger.warning("PrimeKG file %s is empty.", primekg_path)
            self.event_logger.log_event(
                "primekg_empty",
                {
                    "primekg_path": str(primekg_path),
                    "backend_used": "empty_graph",
                    "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                },
            )
            return

        relation_choices = set(get_args(PrimeKGRelation))
        head_id_column = self._resolve_column(frame, ("x_id", "source_id", "head_id"))
        tail_id_column = self._resolve_column(frame, ("y_id", "target_id", "tail_id"))
        head_name_column = self._resolve_column(frame, ("x_name", "source_name", "head_name"))
        tail_name_column = self._resolve_column(frame, ("y_name", "target_name", "tail_name"))
        head_type_column = self._resolve_column(frame, ("x_type", "source_type", "head_type"))
        tail_type_column = self._resolve_column(frame, ("y_type", "target_type", "tail_type"))
        relation_column = self._resolve_column(frame, ("relation", "relation_type"))
        display_relation_column = self._resolve_column(frame, ("display_relation", "relation_display"))
        source_column = self._resolve_column(frame, ("source", "resource", "provenance"), required=False)
        pmid_column = self._resolve_column(frame, ("supporting_pmids", "pubmed_ids", "pmids"), required=False)
        edge_id_column = self._resolve_column(frame, ("edge_id", "id"), required=False)
        head_cui_column = self._resolve_column(frame, ("x_cui", "source_cui", "head_cui"), required=False)
        tail_cui_column = self._resolve_column(frame, ("y_cui", "target_cui", "tail_cui"), required=False)

        for row in frame.itertuples(index=True):
            relation = str(getattr(row, relation_column))
            if relation not in relation_choices:
                continue
            head_id = self._string_or_fallback(getattr(row, head_id_column), fallback=f"node:{row.Index}:head")
            tail_id = self._string_or_fallback(getattr(row, tail_id_column), fallback=f"node:{row.Index}:tail")
            edge_id = (
                self._string_or_fallback(getattr(row, edge_id_column), fallback=f"{head_id}|{relation}|{tail_id}|{row.Index}")
                if edge_id_column is not None
                else f"{head_id}|{relation}|{tail_id}|{row.Index}"
            )
            head_name = self._string_or_fallback(getattr(row, head_name_column), fallback=head_id)
            tail_name = self._string_or_fallback(getattr(row, tail_name_column), fallback=tail_id)
            head_type = self._string_or_fallback(getattr(row, head_type_column), fallback="other")
            tail_type = self._string_or_fallback(getattr(row, tail_type_column), fallback="other")
            display_relation = (
                self._string_or_fallback(getattr(row, display_relation_column), fallback=relation.replace("_", " "))
                if display_relation_column is not None
                else relation.replace("_", " ")
            )
            source_name = (
                self._string_or_fallback(getattr(row, source_column), fallback="primekg") if source_column is not None else "primekg"
            )
            supporting_pmids = (
                self._parse_supporting_pmids(getattr(row, pmid_column)) if pmid_column is not None else []
            )

            source_reliability = self.config.source_reliability_lookup.get(
                source_name.strip().lower(),
                self.config.default_source_reliability,
            )
            edge = KGEdge(
                edge_id=edge_id,
                head=head_id,
                tail=tail_id,
                relation=relation,
                display_relation=display_relation,
                source_reliability=source_reliability,
                amg_confidence=1.0,
                supporting_pmids=supporting_pmids,
            )
            self.edge_lookup[edge.edge_id] = edge
            self.graph.add_node(head_id, name=head_name, node_type=head_type)
            self.graph.add_node(tail_id, name=tail_name, node_type=tail_type)
            self.graph.add_edge(head_id, tail_id, key=edge.edge_id, edge_id=edge.edge_id)

            self.node_name_index[self._normalize_text(head_name)].add(head_id)
            self.node_name_index[self._normalize_text(tail_name)].add(tail_id)
            if head_cui_column is not None:
                head_cui = self._normalize_text(self._string_or_fallback(getattr(row, head_cui_column), fallback=""))
                if head_cui:
                    self.node_cui_index[head_cui].add(head_id)
            if tail_cui_column is not None:
                tail_cui = self._normalize_text(self._string_or_fallback(getattr(row, tail_cui_column), fallback=""))
                if tail_cui:
                    self.node_cui_index[tail_cui].add(tail_id)

        self.node_degrees = {str(node_id): int(degree) for node_id, degree in self.graph.degree()}
        relation_counter = Counter(edge.relation for edge in self.edge_lookup.values())
        self.relation_type_mapping = {relation: index for index, relation in enumerate(sorted(relation_counter))}
        self.backend_used = "networkx_csv"
        self.event_logger.log_event(
            "primekg_loaded",
            {
                "primekg_path": str(primekg_path),
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges(),
                "backend_used": self.backend_used,
                "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            },
        )

    def _load_primekg_from_edge_node_tables(self, edges_path: Path, nodes_path: Path, *, started_at: float) -> bool:
        try:
            nodes_frame = pd.read_csv(nodes_path)
            node_index_column = self._resolve_column(nodes_frame, ("node_index", "index", "id"))
            node_id_column = self._resolve_column(nodes_frame, ("node_id", "id", "primekg_node_id"))
            node_name_column = self._resolve_column(nodes_frame, ("node_name", "name"))
            node_type_column = self._resolve_column(nodes_frame, ("node_type", "type"))
            node_source_column = self._resolve_column(nodes_frame, ("node_source", "source"), required=False)
            node_records: dict[str, dict[str, str]] = {}
            for row in nodes_frame.itertuples(index=False):
                node_index = self._node_index_key(getattr(row, node_index_column))
                node_id = self._string_or_fallback(getattr(row, node_id_column), fallback=f"node:{node_index}")
                node_records[node_index] = {
                    "id": node_id,
                    "name": self._string_or_fallback(getattr(row, node_name_column), fallback=node_id),
                    "type": self._string_or_fallback(getattr(row, node_type_column), fallback="other"),
                    "source": (
                        self._string_or_fallback(getattr(row, node_source_column), fallback="primekg")
                        if node_source_column is not None
                        else "primekg"
                    ),
                }

            edges_frame = pd.read_csv(edges_path)
            relation_choices = set(get_args(PrimeKGRelation))
            head_index_column = self._resolve_column(edges_frame, ("x_index", "source_index", "head_index"))
            tail_index_column = self._resolve_column(edges_frame, ("y_index", "target_index", "tail_index"))
            relation_column = self._resolve_column(edges_frame, ("relation", "relation_type"))
            display_relation_column = self._resolve_column(edges_frame, ("display_relation", "relation_display"))
            edge_id_column = self._resolve_column(edges_frame, ("edge_id", "id"), required=False)
            loaded_edges = 0
            for row in edges_frame.itertuples(index=True):
                relation = str(getattr(row, relation_column))
                if relation not in relation_choices:
                    continue
                head_record = node_records.get(self._node_index_key(getattr(row, head_index_column)))
                tail_record = node_records.get(self._node_index_key(getattr(row, tail_index_column)))
                if head_record is None or tail_record is None:
                    continue
                head_id = head_record["id"]
                tail_id = tail_record["id"]
                edge_id = (
                    self._string_or_fallback(getattr(row, edge_id_column), fallback=f"{head_id}|{relation}|{tail_id}|{row.Index}")
                    if edge_id_column is not None
                    else f"{head_id}|{relation}|{tail_id}|{row.Index}"
                )
                source_name = head_record.get("source") or tail_record.get("source") or "primekg"
                edge = KGEdge(
                    edge_id=edge_id,
                    head=head_id,
                    tail=tail_id,
                    relation=relation,
                    display_relation=self._string_or_fallback(
                        getattr(row, display_relation_column),
                        fallback=relation.replace("_", " "),
                    ),
                    source_reliability=self.config.source_reliability_lookup.get(
                        source_name.strip().lower(),
                        self.config.default_source_reliability,
                    ),
                    amg_confidence=1.0,
                    supporting_pmids=[],
                )
                self.edge_lookup[edge.edge_id] = edge
                self.graph.add_node(head_id, name=head_record["name"], node_type=head_record["type"])
                self.graph.add_node(tail_id, name=tail_record["name"], node_type=tail_record["type"])
                self.graph.add_edge(head_id, tail_id, key=edge.edge_id, edge_id=edge.edge_id)
                self.node_name_index[self._normalize_text(head_record["name"])].add(head_id)
                self.node_name_index[self._normalize_text(tail_record["name"])].add(tail_id)
                loaded_edges += 1

            self.node_degrees = {str(node_id): int(degree) for node_id, degree in self.graph.degree()}
            relation_counter = Counter(edge.relation for edge in self.edge_lookup.values())
            self.relation_type_mapping = {relation: index for index, relation in enumerate(sorted(relation_counter))}
            self.backend_used = "networkx_edge_node_csv"
            self.event_logger.log_event(
                "primekg_loaded",
                {
                    "primekg_path": str(edges_path.parent),
                    "nodes_path": str(nodes_path),
                    "edges_path": str(edges_path),
                    "nodes": self.graph.number_of_nodes(),
                    "edges": loaded_edges,
                    "backend_used": self.backend_used,
                    "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                },
            )
            return loaded_edges > 0
        except Exception as exc:
            self.event_logger.log_exception(
                "primekg_edge_node_load_failed",
                exc,
                {"primekg_path": str(edges_path.parent), "backend_used": "networkx_csv", "latency_ms": 0.0},
            )
            self.graph.clear()
            self.edge_lookup.clear()
            self.node_name_index.clear()
            self.node_cui_index.clear()
            return False

    def _resolve_primekg_edge_node_tables(self, primekg_path: Path) -> tuple[Path, Path] | None:
        if primekg_path.is_file() or not primekg_path.exists():
            return None
        edges_path = primekg_path / "edges.csv"
        nodes_path = primekg_path / "nodes.csv"
        if not edges_path.exists() or not nodes_path.exists():
            return None
        try:
            edge_columns = set(pd.read_csv(edges_path, nrows=0).columns)
            node_columns = set(pd.read_csv(nodes_path, nrows=0).columns)
        except Exception:
            return None
        edge_required = (
            ("x_index", "source_index", "head_index"),
            ("y_index", "target_index", "tail_index"),
            ("relation", "relation_type"),
            ("display_relation", "relation_display"),
        )
        node_required = (
            ("node_index", "index", "id"),
            ("node_id", "id", "primekg_node_id"),
            ("node_name", "name"),
            ("node_type", "type"),
        )
        if all(any(column in edge_columns for column in group) for group in edge_required) and all(
            any(column in node_columns for column in group) for group in node_required
        ):
            return edges_path, nodes_path
        return None

    def _resolve_primekg_file(self, primekg_path: Path) -> Path | None:
        if primekg_path.is_file():
            return primekg_path if self._has_primekg_edge_columns(primekg_path) else None
        if not primekg_path.exists():
            return None
        prioritized = [primekg_path / name for name in PRIMEKG_FILE_PRIORITY if (primekg_path / name).exists()]
        globbed = sorted(
            path
            for pattern in ("primekg*.csv", "*.csv")
            for path in primekg_path.glob(pattern)
            if path.name.lower() not in {"dataset-metadata.json", "datasets-metadata.json"}
        )
        seen: set[Path] = set()
        candidates: list[Path] = []
        for candidate in [*prioritized, *globbed]:
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
        for candidate in candidates:
            if self._has_primekg_edge_columns(candidate):
                return candidate
        self.event_logger.log_event(
            "primekg_schema_mismatch",
            {
                "primekg_path": str(primekg_path),
                "candidate_files": [str(candidate) for candidate in candidates[:20]],
                "backend_used": "empty_graph",
                "latency_ms": 0.0,
            },
        )
        return None

    def _has_primekg_edge_columns(self, csv_path: Path) -> bool:
        try:
            columns = set(pd.read_csv(csv_path, nrows=0).columns)
        except Exception as exc:
            self.event_logger.log_exception(
                "primekg_header_read_failed",
                exc,
                {"primekg_path": str(csv_path), "backend_used": "empty_graph", "latency_ms": 0.0},
            )
            return False
        required_column_groups = (
            ("x_id", "source_id", "head_id"),
            ("y_id", "target_id", "tail_id"),
            ("x_name", "source_name", "head_name"),
            ("y_name", "target_name", "tail_name"),
            ("x_type", "source_type", "head_type"),
            ("y_type", "target_type", "tail_type"),
            ("relation", "relation_type"),
            ("display_relation", "relation_display"),
        )
        return all(any(column in columns for column in group) for group in required_column_groups)

    def _iter_incident_edges(self, node_id: str, allowed_relations: set[str] | None) -> Iterable[KGEdge]:
        for _, target_id, edge_key in self.graph.out_edges(node_id, keys=True):
            edge = self.edge_lookup[edge_key]
            if self._relation_allowed(edge=edge, allowed_relations=allowed_relations):
                yield edge
        for source_id, _, edge_key in self.graph.in_edges(node_id, keys=True):
            edge = self.edge_lookup[edge_key]
            if self._relation_allowed(edge=edge, allowed_relations=allowed_relations):
                yield edge

    def _relation_allowed(self, edge: KGEdge, allowed_relations: set[str] | None) -> bool:
        if allowed_relations is None or not allowed_relations:
            return True
        return edge.relation.lower() in allowed_relations or edge.display_relation.lower() in allowed_relations

    def _compute_personalized_pagerank(self, node_ids: set[str], seed_node_ids: set[str]) -> dict[str, float]:
        if not node_ids:
            return {}
        subgraph = nx.DiGraph()
        for edge in self.edge_lookup.values():
            if edge.head not in node_ids or edge.tail not in node_ids:
                continue
            current_weight = subgraph.get_edge_data(edge.head, edge.tail, {}).get("weight", 0.0)
            subgraph.add_edge(edge.head, edge.tail, weight=current_weight + 1.0)

        if not subgraph.nodes:
            return {}
        personalization = {node_id: 0.0 for node_id in subgraph.nodes}
        active_seed_nodes = [node_id for node_id in seed_node_ids if node_id in personalization]
        if not active_seed_nodes:
            active_seed_nodes = list(subgraph.nodes)
        restart_mass = 1.0 / max(len(active_seed_nodes), 1)
        for node_id in active_seed_nodes:
            personalization[node_id] = restart_mass
        return nx.pagerank(subgraph, alpha=self.config.ppr_alpha, personalization=personalization, weight="weight")

    def _score_edge(self, edge: KGEdge, ppr_scores: dict[str, float], seed_node_ids: set[str]) -> float:
        head_score = ppr_scores.get(edge.head, 0.0)
        tail_score = ppr_scores.get(edge.tail, 0.0)
        seed_bonus = self.config.edge_seed_bonus if edge.head in seed_node_ids or edge.tail in seed_node_ids else 0.0
        return (
            (self.config.edge_node_weight * head_score)
            + (self.config.edge_node_weight * tail_score)
            + (self.config.edge_source_weight * edge.source_reliability)
            + seed_bonus
        )

    def _select_best_node(self, candidate_ids: set[str], entity: QuestionEntity) -> str | None:
        if not candidate_ids:
            return None
        scored_candidates = []
        for candidate_id in candidate_ids:
            node_attributes = self.graph.nodes[candidate_id]
            node_type = self._normalize_text(str(node_attributes.get("node_type", "")))
            node_degree = self.node_degrees.get(candidate_id, 0)
            type_bonus = 1 if self._entity_matches_node_type(entity.entity_type, node_type) else 0
            scored_candidates.append((type_bonus, node_degree, candidate_id))
        scored_candidates.sort(reverse=True)
        return scored_candidates[0][2]

    def _entity_matches_node_type(self, entity_type: str, node_type: str) -> bool:
        normalized_node_type = NODE_TYPE_TO_ENTITY_TYPE.get(node_type, "other")
        return entity_type == normalized_node_type or entity_type == "other"

    @staticmethod
    def _resolve_column(frame: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> str | None:
        for column_name in candidates:
            if column_name in frame.columns:
                return column_name
        if required:
            raise ValueError(f"PrimeKG file is missing required columns: {candidates}")
        return None

    @staticmethod
    def _parse_supporting_pmids(value: Any) -> list[str]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        parts = re.split(r"[|,; ]+", str(value))
        return [part for part in parts if part.isdigit()]

    @staticmethod
    def _normalize_text(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower())

    @staticmethod
    def _string_or_fallback(value: Any, fallback: str) -> str:
        if value is None:
            return fallback
        if isinstance(value, float) and math.isnan(value):
            return fallback
        normalized = str(value).strip()
        return normalized or fallback

    @staticmethod
    def _node_index_key(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and math.isnan(value):
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip()


class PubMedRetriever:
    """PubMed retrieval with BM25 and optional MedCPT reranking."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_path: Path | None = None,
        client: PubMedClient | None = None,
    ) -> None:
        resolved_cache_path = cache_path or DEFAULT_PUBMED_CACHE_PATH
        resolved_cache_path = KaggleEnv.ensure_writeable(resolved_cache_path if resolved_cache_path.is_absolute() else KaggleEnv.path(resolved_cache_path))
        self.config = PubMedRetrieverConfig(api_key=api_key, cache_path=resolved_cache_path)
        self.client = client or PubMedClient(api_key=api_key, cache_path=resolved_cache_path)
        self._query_encoder = None
        self._article_encoder = None
        self._cross_encoder = None
        self.last_extracted_triples: dict[str, list[tuple[str, str, str]]] = {}
        self.event_logger = StructuredLogger("layer1_retrieval", DEFAULT_LOG_DIR)

    @property
    def backend_used(self) -> str:
        if self._query_encoder is not None and self._article_encoder is not None:
            if self._cross_encoder is not None:
                return "bm25+medcpt+rerank"
            return "bm25+medcpt"
        return "bm25_only"

    def retrieve(self, query: str, k: int = DEFAULT_PUBMED_TOP_K) -> list[PubMedPassage]:
        """Retrieve PubMed passages backed by real PMIDs."""

        started_at = time.perf_counter()
        articles = self.client.search_and_fetch(query=query, retmax=self.config.search_candidate_count)
        if not articles:
            self.event_logger.log_event(
                "pubmed_retrieved",
                {
                    "query": query,
                    "k_requested": k,
                    "k_returned": 0,
                    "backend_used": self.backend_used,
                    "latency_ms": (time.perf_counter() - started_at) * 1000.0,
                },
            )
            return []

        document_texts = [self._article_to_text(article) for article in articles]
        bm25_scores = self._bm25_scores(query=query, documents=document_texts)
        dense_scores = self._dense_scores(query=query, documents=document_texts)
        candidate_pool_size = min(self.config.cross_encoder_pool, len(articles))
        cross_encoder_scores = self._cross_encoder_scores(
            query=query,
            documents=document_texts,
            candidate_indices=np.argsort(-(bm25_scores + dense_scores))[:candidate_pool_size],
        )
        combined_scores = self._combine_scores(bm25_scores=bm25_scores, dense_scores=dense_scores, cross_encoder_scores=cross_encoder_scores)

        ranked_indices = np.argsort(-combined_scores)[:k]
        passages: list[PubMedPassage] = []
        self.last_extracted_triples = {}
        for index in ranked_indices:
            article = articles[int(index)]
            self.last_extracted_triples[article.pmid] = self._extract_candidate_triples(article.abstract)
            passages.append(
                PubMedPassage(
                    pmid=article.pmid,
                    title=article.title,
                    abstract=article.abstract,
                    relevance_score=float(combined_scores[int(index)]),
                )
            )
        self.event_logger.log_event(
            "pubmed_retrieved",
            {
                "query": query,
                "k_requested": k,
                "k_returned": len(passages),
                "backend_used": self.backend_used,
                "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            },
        )
        return passages

    def _bm25_scores(self, query: str, documents: Sequence[str]) -> np.ndarray:
        tokenized_documents = [self._tokenize(text) for text in documents]
        if not any(tokenized_documents):
            return np.zeros(len(documents), dtype=np.float32)
        bm25 = BM25Okapi(tokenized_documents)
        raw_scores = bm25.get_scores(self._tokenize(query))
        return EntityLinker._normalize_scores(raw_scores)

    def _dense_scores(self, query: str, documents: Sequence[str]) -> np.ndarray:
        if not documents or self.config.query_encoder_model_name is None or self.config.article_encoder_model_name is None:
            return np.zeros(len(documents), dtype=np.float32)
        encoders = self._load_bi_encoders()
        if encoders is None:
            return np.zeros(len(documents), dtype=np.float32)
        query_encoder, article_encoder = encoders
        query_embedding = query_encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        document_embeddings = article_encoder.encode(list(documents), convert_to_numpy=True, normalize_embeddings=True)
        scores = np.dot(document_embeddings, query_embedding)
        return EntityLinker._normalize_scores(scores)

    def _cross_encoder_scores(
        self,
        query: str,
        documents: Sequence[str],
        candidate_indices: Sequence[int],
    ) -> np.ndarray:
        scores = np.zeros(len(documents), dtype=np.float32)
        if not documents or self.config.cross_encoder_model_name is None:
            return scores
        cross_encoder = self._load_cross_encoder()
        if cross_encoder is None:
            return scores
        candidate_indices = [int(index) for index in candidate_indices]
        if not candidate_indices:
            return scores
        pairs = [[query, documents[index]] for index in candidate_indices]
        predictions = np.asarray(cross_encoder.predict(pairs), dtype=np.float32)
        normalized_predictions = EntityLinker._normalize_scores(predictions)
        for index, score in zip(candidate_indices, normalized_predictions):
            scores[index] = score
        return scores

    def _combine_scores(
        self,
        bm25_scores: np.ndarray,
        dense_scores: np.ndarray,
        cross_encoder_scores: np.ndarray,
    ) -> np.ndarray:
        weighted_components: list[tuple[float, np.ndarray]] = []
        if np.any(bm25_scores):
            weighted_components.append((self.config.bm25_weight, bm25_scores))
        if np.any(dense_scores):
            weighted_components.append((self.config.dense_weight, dense_scores))
        if np.any(cross_encoder_scores):
            weighted_components.append((self.config.cross_encoder_weight, cross_encoder_scores))

        if not weighted_components:
            return bm25_scores

        total_weight = sum(weight for weight, _ in weighted_components)
        combined_scores = np.zeros_like(bm25_scores, dtype=np.float32)
        for weight, component in weighted_components:
            combined_scores += (weight / total_weight) * component
        return combined_scores

    def _load_bi_encoders(self) -> tuple[Any, Any] | None:
        if self._query_encoder is not None and self._article_encoder is not None:
            return self._query_encoder, self._article_encoder
        try:
            from sentence_transformers import SentenceTransformer

            self._query_encoder = SentenceTransformer(self.config.query_encoder_model_name)
            self._article_encoder = SentenceTransformer(self.config.article_encoder_model_name)
        except Exception as exc:
            logger.warning("Unable to load MedCPT bi-encoders. Falling back to BM25-only PubMed ranking: %s", exc)
            self.event_logger.log_exception(
                "pubmed_dense_fallback",
                exc,
                {"backend_used": "bm25_only", "latency_ms": 0.0},
            )
            self.config.query_encoder_model_name = None
            self.config.article_encoder_model_name = None
            return None
        return self._query_encoder, self._article_encoder

    def _load_cross_encoder(self) -> Any | None:
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.config.cross_encoder_model_name)
        except Exception as exc:
            logger.warning("Unable to load MedCPT cross-encoder. Continuing without reranking: %s", exc)
            self.event_logger.log_exception(
                "pubmed_rerank_fallback",
                exc,
                {"backend_used": self.backend_used, "latency_ms": 0.0},
            )
            self.config.cross_encoder_model_name = None
            return None
        return self._cross_encoder

    def _extract_candidate_triples(self, text: str) -> list[tuple[str, str, str]]:
        triples: list[tuple[str, str, str]] = []
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            matches = re.finditer(
                r"([A-Za-z0-9-][A-Za-z0-9\- ]+?)\s+(reduces|increases|causes|treats|prevents|inhibits|activates)\s+([A-Za-z0-9-][A-Za-z0-9\- ]+)",
                sentence,
                flags=re.IGNORECASE,
            )
            for match in matches:
                subject, relation, obj = match.groups()
                triples.append((subject.strip(), relation.lower(), obj.strip()))
        return triples

    @staticmethod
    def _article_to_text(article: PubMedArticle) -> str:
        return " ".join(part for part in [article.title, article.abstract] if part).strip()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[A-Za-z0-9-]+", text.lower())


class AgenticRetriever:
    """Orchestrates entity linking, graph extraction, and PubMed retrieval."""

    def __init__(
        self,
        primekg_path: Path = DEFAULT_PRIMEKG_PATH,
        pubmed_api_key: str | None = None,
        pubmed_cache_path: Path | None = None,
        scispacy_model: str = DEFAULT_SCISPACY_MODEL,
        relation_filter: set[str] | None = None,
        linker: EntityLinker | None = None,
        kg_extractor: PrimeKGExtractor | None = None,
        pubmed: PubMedRetriever | None = None,
    ) -> None:
        resolved_cache_path = pubmed_cache_path or DEFAULT_PUBMED_CACHE_PATH
        resolved_primekg_path = primekg_path if primekg_path.is_absolute() else KaggleEnv.path(primekg_path)
        resolved_cache_path = KaggleEnv.ensure_writeable(resolved_cache_path if resolved_cache_path.is_absolute() else KaggleEnv.path(resolved_cache_path))
        self.config = AgenticRetrieverConfig(
            primekg_path=resolved_primekg_path,
            pubmed_cache_path=resolved_cache_path,
            scispacy_model=scispacy_model,
            relation_filter=relation_filter,
            pubmed_api_key=pubmed_api_key,
        )
        self.linker = linker or EntityLinker(scispacy_model=scispacy_model)
        self.kg_extractor = kg_extractor or PrimeKGExtractor(primekg_path=resolved_primekg_path)
        self.pubmed = pubmed or PubMedRetriever(api_key=pubmed_api_key, cache_path=resolved_cache_path)
        self.event_logger = StructuredLogger("layer1_retrieval", DEFAULT_LOG_DIR)

    def retrieve(self, question: str) -> EvidenceBundle:
        """Transform a raw question into a fully-typed EvidenceBundle."""

        started_at = time.perf_counter()
        question_type = self._classify_question_type(question)

        entity_start = time.perf_counter()
        question_entities = self.linker.link(question)
        question_entities = self.kg_extractor.resolve_seed_entities(question_entities)
        entity_latency_ms = (time.perf_counter() - entity_start) * 1000.0

        kg_start = time.perf_counter()
        relation_filter = self.config.relation_filter or QUESTION_TYPE_RELATION_FILTERS.get(question_type) or None
        subgraph_edges = self.kg_extractor.extract_2hop(
            seed_entities=question_entities,
            top_k=self.config.primekg_top_k,
            relation_filter=relation_filter,
        )
        kg_latency_ms = (time.perf_counter() - kg_start) * 1000.0

        pubmed_start = time.perf_counter()
        pubmed_passages = self.pubmed.retrieve(question, k=self.config.pubmed_top_k)
        pubmed_latency_ms = (time.perf_counter() - pubmed_start) * 1000.0

        total_latency_ms = (time.perf_counter() - started_at) * 1000.0
        metadata = {
            "retrieval_latency_ms": total_latency_ms,
            "entity_linking_latency_ms": entity_latency_ms,
            "primekg_latency_ms": kg_latency_ms,
            "pubmed_latency_ms": pubmed_latency_ms,
            "entity_count": len(question_entities),
            "edge_count": len(subgraph_edges),
            "pubmed_count": len(pubmed_passages),
            "primekg_loaded": bool(self.kg_extractor.graph),
            "pubmed_cache_path": str(self.config.pubmed_cache_path),
            "question_type": question_type,
            "entity_linker_backend": self.linker.backend_used,
            "pubmed_backend": self.pubmed.backend_used,
            "backend_used": {
                "entity_linker": self.linker.backend_used,
                "primekg": self.kg_extractor.backend_used,
                "pubmed": self.pubmed.backend_used,
            },
        }
        self.event_logger.log_event(
            "retrieval_complete",
            {
                "question_type": question_type,
                "entity_count": len(question_entities),
                "edge_count": len(subgraph_edges),
                "pubmed_count": len(pubmed_passages),
                "backend_used": metadata["backend_used"],
                "latency_ms": total_latency_ms,
            },
        )

        return EvidenceBundle(
            question_text=question,
            question_type=question_type,
            question_entities=question_entities,
            subgraph_edges=subgraph_edges,
            pubmed_passages=pubmed_passages,
            metadata=metadata,
        )

    def _classify_question_type(self, question: str) -> QuestionType:
        normalized_question = question.lower()
        for question_type, patterns in QUESTION_TYPE_PATTERNS.items():
            if question_type in {"factoid", "other"}:
                continue
            if any(pattern in normalized_question for pattern in patterns):
                return question_type
        return "factoid"


__all__ = ["AgenticRetriever", "EntityLinker", "PrimeKGExtractor", "PubMedRetriever"]
