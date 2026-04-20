from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from src.layers.layer1_retrieval import PrimeKGExtractor
from src.schemas import EvidenceBundle, QuestionEntity
from src.utils.kaggle_env import KaggleEnv

DEFAULT_PRETRAIN_QUESTION_TYPE = "factoid"
DEFAULT_MAX_SEED_ENTITIES = 3
DEFAULT_MIN_SEED_ENTITIES = 1
DEFAULT_TOP_K = 96
ALLOWED_QUESTION_ENTITY_TYPES = frozenset({"disease", "drug", "protein", "symptom", "anatomy", "other"})


@dataclass(slots=True)
class VQSubgraphSample:
    sample_index: int
    seed_node_ids: list[str]
    evidence: EvidenceBundle


class PrimeKGVQDataset:
    def __init__(
        self,
        *,
        primekg_path: str | Path,
        seed: int = 42,
        top_k: int = DEFAULT_TOP_K,
        min_seed_entities: int = DEFAULT_MIN_SEED_ENTITIES,
        max_seed_entities: int = DEFAULT_MAX_SEED_ENTITIES,
        max_edges_to_load: int | None = None,
    ) -> None:
        resolved_path = primekg_path if isinstance(primekg_path, Path) and primekg_path.is_absolute() else KaggleEnv.path(Path(primekg_path))
        self.primekg_path = resolved_path
        self.seed = int(seed)
        self.top_k = int(top_k)
        self.min_seed_entities = max(1, int(min_seed_entities))
        self.max_seed_entities = max(self.min_seed_entities, int(max_seed_entities))
        self.max_edges_to_load = max_edges_to_load if max_edges_to_load is None else int(max_edges_to_load)
        self.extractor = PrimeKGExtractor(primekg_path=resolved_path, max_edges_to_load=self.max_edges_to_load)
        self.node_ids = sorted(str(node_id) for node_id in self.extractor.graph.nodes)
        if not self.node_ids:
            raise ValueError(f"PrimeKG graph is empty or unavailable at {resolved_path}. Cannot pretrain a VQ codebook.")

    def __len__(self) -> int:
        return len(self.node_ids)

    def sample(self, sample_index: int) -> VQSubgraphSample:
        rng = random.Random(f"{self.seed}:{int(sample_index)}")
        seed_count = rng.randint(self.min_seed_entities, self.max_seed_entities)
        seed_node_ids = self._choose_seed_nodes(rng=rng, count=seed_count)
        evidence = self._build_evidence(seed_node_ids=seed_node_ids, sample_index=int(sample_index))
        return VQSubgraphSample(sample_index=int(sample_index), seed_node_ids=seed_node_ids, evidence=evidence)

    def iter_samples(self, start: int, stop: int) -> Iterator[VQSubgraphSample]:
        for sample_index in range(int(start), int(stop)):
            yield self.sample(sample_index)

    def graph_signature_payload(self) -> dict[str, object]:
        return {
            "primekg_path": str(self.primekg_path),
            "backend_used": self.extractor.backend_used,
            "nodes": int(self.extractor.graph.number_of_nodes()),
            "edges": int(self.extractor.graph.number_of_edges()),
            "max_edges_to_load": self.max_edges_to_load,
        }

    def _choose_seed_nodes(self, *, rng: random.Random, count: int) -> list[str]:
        if count >= len(self.node_ids):
            return list(self.node_ids)
        return sorted(rng.sample(self.node_ids, k=count))

    def _build_evidence(self, *, seed_node_ids: Sequence[str], sample_index: int) -> EvidenceBundle:
        question_entities = [self._question_entity_for_node(node_id) for node_id in seed_node_ids]
        subgraph_edges = self.extractor.extract_2hop(question_entities, top_k=self.top_k, relation_filter=None)
        if not subgraph_edges:
            # PrimeKGExtractor returns an empty list when a seed cannot be expanded.
            # Falling back to a deterministic single-edge neighborhood preserves progress.
            for node_id in seed_node_ids:
                incident_edges = list(self.extractor._iter_incident_edges(node_id=node_id, allowed_relations=None))
                if incident_edges:
                    subgraph_edges = incident_edges[: self.top_k]
                    break
        question_text = f"PrimeKG VQ pretraining sample {sample_index}"
        return EvidenceBundle(
            question_text=question_text,
            question_type=DEFAULT_PRETRAIN_QUESTION_TYPE,
            question_entities=question_entities,
            subgraph_edges=[edge.model_copy(deep=True) for edge in subgraph_edges],
            pubmed_passages=[],
            metadata={
                "sample_index": int(sample_index),
                "seed_node_ids": list(seed_node_ids),
                "primekg_backend": self.extractor.backend_used,
                "primekg_path": str(self.primekg_path),
            },
        )

    def _question_entity_for_node(self, node_id: str) -> QuestionEntity:
        node_payload = dict(self.extractor.graph.nodes.get(node_id, {}))
        node_name = str(node_payload.get("name") or node_id)
        node_type = self._normalize_question_entity_type(str(node_payload.get("node_type") or "other"))
        return QuestionEntity(
            surface=node_name,
            cui=None,
            primekg_node_id=node_id,
            entity_type=node_type,
        )

    @staticmethod
    def _normalize_question_entity_type(value: str) -> str:
        normalized = value.strip().lower()
        if normalized in ALLOWED_QUESTION_ENTITY_TYPES:
            return normalized
        return "other"


__all__ = ["PrimeKGVQDataset", "VQSubgraphSample"]
