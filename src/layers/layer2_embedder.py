from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, get_args

import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn

from src.schemas import EvidenceBundle, PrimeKGRelation, TRMInputBundle
from src.utils.kaggle_env import KaggleEnv
from src.utils.path_resolver import KagglePathResolver
from src.utils.structured_logger import DEFAULT_LOG_DIR, StructuredLogger

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


DEFAULT_TEXT_EMBED_DIM = 768
DEFAULT_EDGE_EMBED_DIM = 64
DEFAULT_HGT_HEADS = 4
DEFAULT_HGT_LAYERS = 2
DEFAULT_SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
DEFAULT_SAPBERT_KAGGLE_PATH = Path("/kaggle/input/medv3-checkpoints/sapbert")
DEFAULT_SAPBERT_LOCAL_PATH = Path("data/checkpoints/sapbert")
DEFAULT_MEDCPT_ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
DEFAULT_VECTOR_QUANTIZE_BETA = 0.25
DEFAULT_PPR_ALPHA = 0.85
DEFAULT_PASSAGE_PREFIX = "PMID:"
DEFAULT_PAD_IDENTIFIER = "__pad__"

QUESTION_TYPE_TO_PUZZLE_ID = {
    "drug_interaction": 0,
    "diagnosis": 1,
    "dosage": 2,
    "etiology": 3,
    "factoid": 4,
    "other": 4,
}

NODE_TYPE_VOCAB = (
    "disease",
    "drug",
    "protein",
    "symptom",
    "anatomy",
    "phenotype",
    "pathway",
    "exposure",
    "bioprocess",
    "other",
)
NODE_TYPE_TO_INDEX = {node_type: index for index, node_type in enumerate(NODE_TYPE_VOCAB)}
RELATION_TO_INDEX = {relation: index for index, relation in enumerate(get_args(PrimeKGRelation))}
ID_PREFIX_TO_NODE_TYPE = {
    "disease": "disease",
    "drug": "drug",
    "protein": "protein",
    "symptom": "symptom",
    "anatomy": "anatomy",
    "phenotype": "phenotype",
    "pathway": "pathway",
    "exposure": "exposure",
    "bioprocess": "bioprocess",
    "cellcomp": "other",
    "molfunc": "other",
    "effect": "other",
}


@dataclass(slots=True)
class GraphInputs:
    node_ids: list[str]
    node_names: list[str]
    node_type_ids: torch.Tensor
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    node_scores: dict[str, float]


class DeterministicHashTextEncoder:
    """Fallback text encoder used when frozen HF checkpoints are unavailable locally."""

    def __init__(self, output_dim: int = DEFAULT_TEXT_EMBED_DIM, salt: str = "medical-text") -> None:
        self.output_dim = output_dim
        self.salt = salt

    def encode(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        embeddings = torch.stack([self._encode_single_text(text) for text in texts], dim=0)
        return embeddings.to(device)

    def _encode_single_text(self, text: str) -> torch.Tensor:
        vector = torch.zeros(self.output_dim, dtype=torch.float32)
        normalized_text = text.strip().lower()
        tokens = re.findall(r"[A-Za-z0-9:-]+", normalized_text)
        if not tokens:
            tokens = ["[empty]"]

        for token_index, token in enumerate(tokens):
            digest = hashlib.sha256(f"{self.salt}:{token_index}:{token}".encode("utf-8")).digest()
            for offset in range(0, len(digest), 4):
                chunk = digest[offset : offset + 4]
                hashed_value = int.from_bytes(chunk, "little", signed=False)
                feature_index = hashed_value % self.output_dim
                magnitude = ((hashed_value >> 8) & 0xFF) / 255.0
                sign = 1.0 if (hashed_value & 1) == 0 else -1.0
                vector[feature_index] += sign * (0.25 + magnitude)

        norm = torch.linalg.norm(vector).clamp_min(1e-6)
        return vector / norm


class FrozenHFTextEncoder:
    """Loads a local HF encoder if present, otherwise falls back to deterministic hashing."""

    def __init__(
        self,
        model_name: str | None,
        output_dim: int = DEFAULT_TEXT_EMBED_DIM,
        local_files_only: bool = True,
        fallback_salt: str = "medical-text",
    ) -> None:
        self.model_name = _resolve_model_artifact(model_name)
        self.output_dim = output_dim
        self.local_files_only = local_files_only
        self.fallback_encoder = DeterministicHashTextEncoder(output_dim=output_dim, salt=fallback_salt)
        self._tokenizer = None
        self._model = None
        self.last_backend_used = "uninitialized"

    @property
    def backend_used(self) -> str:
        if self._tokenizer is not None and self._model is not None:
            return "hf_local"
        if self.model_name is None:
            return "hash_fallback"
        return self.last_backend_used

    def encode(self, texts: Sequence[str], device: torch.device) -> torch.Tensor:
        if not texts:
            if self.model_name is None:
                self.last_backend_used = "hash_fallback"
            elif self._tokenizer is not None and self._model is not None:
                self.last_backend_used = "hf_local"
            else:
                self.last_backend_used = "unused"
            return torch.zeros((0, self.output_dim), dtype=torch.float32, device=device)
        if self.model_name is None:
            self.last_backend_used = "hash_fallback"
            return self.fallback_encoder.encode(texts=texts, device=device)

        model_components = self._load_model()
        if model_components is None:
            self.last_backend_used = "hash_fallback"
            return self.fallback_encoder.encode(texts=texts, device=device)

        tokenizer, model = model_components
        self.last_backend_used = "hf_local"
        encoded_inputs = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(**encoded_inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = encoded_inputs["attention_mask"].unsqueeze(-1)
        pooled = (last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1)
        if pooled.shape[-1] != self.output_dim:
            if pooled.shape[-1] > self.output_dim:
                pooled = pooled[:, : self.output_dim]
            else:
                pooled = F.pad(pooled, (0, self.output_dim - pooled.shape[-1]))
        return pooled.to(device=device, dtype=torch.float32)

    def _load_model(self) -> tuple[Any, Any] | None:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=self.local_files_only)
            self._model = AutoModel.from_pretrained(self.model_name, local_files_only=self.local_files_only)
            self._model.requires_grad_(False)
            hidden_size = int(getattr(getattr(self._model, "config", None), "hidden_size", self.output_dim))
            logger.info("Loaded local HF encoder from %s (%sd).", self.model_name, hidden_size)
        except Exception as exc:
            logger.warning("Falling back to deterministic text encoder because %s could not be loaded: %s", self.model_name, exc)
            self.model_name = None
            self._tokenizer = None
            self._model = None
            return None
        return self._tokenizer, self._model


class SimpleVectorQuantize(nn.Module):
    """Nearest-neighbor vector quantizer with a reserved padding token."""

    def __init__(self, dim: int, codebook_size: int, beta: float = DEFAULT_VECTOR_QUANTIZE_BETA) -> None:
        super().__init__()
        if codebook_size < 2:
            raise ValueError("codebook_size must be at least 2 so one token can be reserved for padding.")
        self.dim = dim
        self.codebook_size = codebook_size
        self.beta = beta
        self.padding_index = codebook_size - 1
        self.codebook = nn.Embedding(codebook_size - 1, dim)
        self._initialize_codebook()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inputs.ndim != 2:
            raise ValueError("Vector quantizer expects [num_items, dim] inputs.")
        if inputs.size(0) == 0:
            empty_indices = torch.empty((0,), dtype=torch.long, device=inputs.device)
            empty_quantized = torch.empty_like(inputs)
            zero_loss = torch.zeros((), dtype=inputs.dtype, device=inputs.device)
            return empty_quantized, empty_indices, zero_loss

        codebook = self.codebook.weight
        distances = torch.cdist(inputs, codebook)
        indices = torch.argmin(distances, dim=-1)
        quantized = F.embedding(indices, codebook)
        commitment_loss = F.mse_loss(inputs, quantized.detach()) + self.beta * F.mse_loss(quantized, inputs.detach())
        quantized = inputs + (quantized - inputs).detach()
        return quantized, indices.to(dtype=torch.long), commitment_loss

    def _initialize_codebook(self) -> None:
        with torch.no_grad():
            positions = torch.arange(self.codebook.num_embeddings, dtype=torch.float32).unsqueeze(1)
            dimensions = torch.arange(self.dim, dtype=torch.float32).unsqueeze(0)
            weights = torch.sin(positions * 0.017 + dimensions * 0.013) + torch.cos(positions * 0.007 - dimensions * 0.019)
            weights = F.normalize(weights, dim=-1)
            self.codebook.weight.copy_(weights)


class FallbackHGTLayer(nn.Module):
    """Lightweight relation-aware graph transformer used when torch_geometric is unavailable."""

    def __init__(self, hidden_dim: int, edge_dim: int, n_node_types: int, heads: int) -> None:
        super().__init__()
        self.node_type_emb = nn.Embedding(n_node_types, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if node_states.ndim != 2:
            raise ValueError("node_states must be [num_nodes, hidden_dim].")

        updated_states = node_states + self.node_type_emb(node_type_ids)
        if edge_index.numel() > 0:
            source_indices = edge_index[0]
            target_indices = edge_index[1]
            relation_messages = self.edge_proj(edge_attr)
            aggregated_messages = torch.zeros_like(updated_states)
            message_counts = torch.zeros((updated_states.size(0), 1), dtype=updated_states.dtype, device=updated_states.device)
            aggregated_messages.index_add_(0, target_indices, updated_states[source_indices] + relation_messages)
            message_counts.index_add_(
                0,
                target_indices,
                torch.ones((target_indices.size(0), 1), dtype=updated_states.dtype, device=updated_states.device),
            )
            aggregated_messages = aggregated_messages / message_counts.clamp_min(1.0)
            updated_states = updated_states + aggregated_messages

        attn_input = updated_states.unsqueeze(0)
        attn_output, _ = self.self_attn(attn_input, attn_input, attn_input, need_weights=False)
        updated_states = self.norm1(updated_states + attn_output.squeeze(0))
        feed_forward = self.ffn(updated_states)
        return self.norm2(updated_states + feed_forward)


class MedicalGraphEmbedder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        codebook_size: int = 4096,
        max_len: int = 256,
        n_relations: int = 30,
        n_node_types: int = 10,
        sapbert_model_name: str | None = DEFAULT_SAPBERT_MODEL,
        medcpt_article_model_name: str | None = DEFAULT_MEDCPT_ARTICLE_MODEL,
        local_files_only: bool = True,
        device: str | torch.device | None = None,
        sapbert_path: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.max_len = max_len
        self.n_relations = n_relations
        self.n_node_types = n_node_types
        self.device = torch.device(device or "cpu")
        self.event_logger = StructuredLogger("layer2_embedder", DEFAULT_LOG_DIR)

        self.node_proj = nn.Linear(DEFAULT_TEXT_EMBED_DIM + n_node_types + 2, hidden_dim)
        self.edge_emb = nn.Embedding(n_relations, DEFAULT_EDGE_EMBED_DIM)
        self.edge_proj = nn.Linear(DEFAULT_EDGE_EMBED_DIM + 2, DEFAULT_EDGE_EMBED_DIM)
        self.hgt = nn.ModuleList(
            [
                FallbackHGTLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=DEFAULT_EDGE_EMBED_DIM,
                    n_node_types=n_node_types,
                    heads=DEFAULT_HGT_HEADS,
                )
                for _ in range(DEFAULT_HGT_LAYERS)
            ]
        )
        self.passage_proj = nn.Linear(DEFAULT_TEXT_EMBED_DIM, hidden_dim)
        self.bridge_attn = nn.MultiheadAttention(hidden_dim, DEFAULT_HGT_HEADS, batch_first=True)
        self.vq = SimpleVectorQuantize(dim=hidden_dim, codebook_size=codebook_size)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        resolved_sapbert_path = get_sapbert_path(sapbert_path) if sapbert_path is not None else None
        effective_sapbert_model = resolved_sapbert_path or sapbert_model_name

        self.node_text_encoder = FrozenHFTextEncoder(
            model_name=effective_sapbert_model,
            output_dim=DEFAULT_TEXT_EMBED_DIM,
            local_files_only=local_files_only,
            fallback_salt="sapbert",
        )
        self.passage_text_encoder = FrozenHFTextEncoder(
            model_name=medcpt_article_model_name,
            output_dim=DEFAULT_TEXT_EMBED_DIM,
            local_files_only=local_files_only,
            fallback_salt="medcpt-article",
        )

        self.relation_to_index = dict(RELATION_TO_INDEX)
        self.node_type_to_index = {node_type: index for index, node_type in enumerate(NODE_TYPE_VOCAB[:n_node_types])}
        self.padding_token_id = self.vq.padding_index
        self.backend_used = "fallback_hash"
        self.last_backend_used: dict[str, str] = {}
        self._initialize_sapbert_backend()
        self.to(self.device)
        self.eval()

    def forward(self, bundle: EvidenceBundle) -> dict[str, Any]:
        graph_inputs = self._build_graph_inputs(bundle)
        if graph_inputs.node_ids:
            node_hidden = self.node_proj(graph_inputs.node_features.to(self.device))
            node_type_ids = graph_inputs.node_type_ids.to(self.device)
            edge_index = graph_inputs.edge_index.to(self.device)
            raw_edge_features = graph_inputs.edge_features.to(self.device)
            edge_features = self.edge_proj(raw_edge_features) if raw_edge_features.numel() > 0 else raw_edge_features
            for layer in self.hgt:
                node_hidden = layer(node_hidden, edge_index, edge_features, node_type_ids)
        else:
            node_hidden = torch.zeros((0, self.hidden_dim), dtype=torch.float32, device=self.device)

        passage_hidden, passage_ids, passage_scores = self._encode_passages(bundle, node_hidden)
        combined_embeddings, combined_mapping = self._rank_and_select(
            node_ids=graph_inputs.node_ids,
            node_embeddings=node_hidden,
            node_scores=graph_inputs.node_scores,
            passage_ids=passage_ids,
            passage_embeddings=passage_hidden,
            passage_scores=passage_scores,
        )
        token_ids, codebook_indices, node_mapping = self._quantize_selection(combined_embeddings, combined_mapping)
        puzzle_identifiers = torch.tensor(
            [QUESTION_TYPE_TO_PUZZLE_ID.get(bundle.question_type, QUESTION_TYPE_TO_PUZZLE_ID["other"])],
            dtype=torch.long,
            device=self.device,
        )
        node_backend = self.node_text_encoder.backend_used
        passage_backend = self.passage_text_encoder.backend_used
        if node_backend == "uninitialized" and not graph_inputs.node_ids:
            node_backend = "unused"
        if passage_backend == "uninitialized" and not bundle.pubmed_passages:
            passage_backend = "unused"
        self.backend_used = "sapbert_real" if node_backend == "hf_local" else "fallback_hash"
        graph_backend = "sapbert_hgt" if self.backend_used == "sapbert_real" else "fallback_hgt"
        self.last_backend_used = {
            "node_text_encoder": node_backend,
            "passage_text_encoder": passage_backend,
            "graph_encoder": graph_backend,
            "vector_quantizer": "deterministic_vq",
            "layer2_backend": self.backend_used,
        }
        self.event_logger.log_event(
            "graph_encoded",
            {
                "encoder": node_backend,
                "vq_tokens": int(token_ids.numel()),
                "layer2_backend": self.backend_used,
                "backend_used": dict(self.last_backend_used),
                "latency_ms": None,
            },
        )
        self.event_logger.log_event(
            "vq_lookup",
            {
                "codebook_hits": int((codebook_indices != self.padding_token_id).sum().item()),
                "pad_tokens": int((codebook_indices == self.padding_token_id).sum().item()),
                "backend_used": "deterministic_vq",
                "latency_ms": None,
            },
        )
        return {
            "inputs": token_ids.unsqueeze(0),
            "puzzle_identifiers": puzzle_identifiers,
            "codebook_indices": codebook_indices,
            "node_mapping": node_mapping,
            "layer2_backend": self.backend_used,
            "backend_used": dict(self.last_backend_used),
        }

    def to_trm_input_bundle(self, bundle: EvidenceBundle) -> TRMInputBundle:
        encoded = self.forward(bundle)
        return TRMInputBundle(
            inputs=encoded["inputs"],
            puzzle_identifiers=encoded["puzzle_identifiers"],
            original_graph=bundle,
        )

    def pretrain_vq(
        self,
        subgraph_samples: Iterable[Any],
        epochs: int = 1,
        lr: float = 1e-3,
    ) -> list[dict[str, float]]:
        samples = list(subgraph_samples)
        if not samples:
            return []

        was_training = self.training
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history: list[dict[str, float]] = []

        for epoch_index in range(epochs):
            total_loss = 0.0
            total_recon_loss = 0.0
            total_commit_loss = 0.0
            sample_count = 0

            for sample in samples:
                target_embeddings = self._extract_pretraining_embeddings(sample)
                if target_embeddings is None or target_embeddings.numel() == 0:
                    continue
                target_embeddings = target_embeddings.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                quantized, _, commit_loss = self.vq(target_embeddings)
                reconstruction = self.reconstruction_head(quantized)
                reconstruction_loss = F.mse_loss(reconstruction, target_embeddings)
                loss = reconstruction_loss + commit_loss
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                total_recon_loss += float(reconstruction_loss.detach().cpu())
                total_commit_loss += float(commit_loss.detach().cpu())
                sample_count += 1

            if sample_count == 0:
                history.append({"epoch": float(epoch_index), "loss": 0.0, "reconstruction_loss": 0.0, "commit_loss": 0.0})
            else:
                history.append(
                    {
                        "epoch": float(epoch_index),
                        "loss": total_loss / sample_count,
                        "reconstruction_loss": total_recon_loss / sample_count,
                        "commit_loss": total_commit_loss / sample_count,
                    }
                )
            self.event_logger.log_metric("vq_pretrain_loss", history[-1]["loss"], step=epoch_index)

        self.train(was_training)
        if not was_training:
            self.eval()
        return history

    def _initialize_sapbert_backend(self) -> None:
        if self.node_text_encoder.model_name is None:
            self.backend_used = "fallback_hash"
            logger.warning("SapBERT model path is not configured; fallback hash encoder will be used.")
            return
        try:
            _ = self.node_text_encoder.encode(["sapbert warmup"], device=self.device)
            if self.node_text_encoder.backend_used == "hf_local":
                self.backend_used = "sapbert_real"
                logger.info("SapBERT loaded successfully from %s.", self.node_text_encoder.model_name)
            else:
                self.backend_used = "fallback_hash"
                logger.warning("SapBERT warmup fell back to hash backend.")
        except Exception as exc:
            self.backend_used = "fallback_hash"
            logger.error("SapBERT warmup failed: %s", exc)

    def _build_graph_inputs(self, bundle: EvidenceBundle) -> GraphInputs:
        graph = nx.DiGraph()
        node_names: dict[str, str] = {}
        node_types: dict[str, str] = {}
        seed_node_ids: list[str] = []

        for index, entity in enumerate(bundle.question_entities):
            node_id = entity.primekg_node_id or f"entity:{index}:{self._slugify(entity.surface)}"
            graph.add_node(node_id)
            node_names[node_id] = entity.surface
            node_types[node_id] = self._normalize_node_type(entity.entity_type)
            seed_node_ids.append(node_id)

        for edge in bundle.subgraph_edges:
            graph.add_node(edge.head)
            graph.add_node(edge.tail)
            graph.add_edge(edge.head, edge.tail, edge_id=edge.edge_id)
            node_names.setdefault(edge.head, self._humanize_identifier(edge.head))
            node_names.setdefault(edge.tail, self._humanize_identifier(edge.tail))
            node_types.setdefault(edge.head, self._infer_node_type(edge.head))
            node_types.setdefault(edge.tail, self._infer_node_type(edge.tail))

        node_ids = sorted(graph.nodes)
        if not node_ids:
            return GraphInputs(
                node_ids=[],
                node_names=[],
                node_type_ids=torch.zeros((0,), dtype=torch.long),
                node_features=torch.zeros((0, DEFAULT_TEXT_EMBED_DIM + self.n_node_types + 2), dtype=torch.float32),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                edge_features=torch.zeros((0, DEFAULT_EDGE_EMBED_DIM + 2), dtype=torch.float32),
                node_scores={},
            )

        degree_lookup = dict(graph.degree())
        max_degree = max(degree_lookup.values(), default=1)
        node_scores = self._compute_personalized_pagerank(graph=graph, node_ids=node_ids, seed_node_ids=seed_node_ids)
        text_embeddings = self.node_text_encoder.encode([node_names[node_id] for node_id in node_ids], device=self.device).cpu()

        node_feature_rows = []
        node_type_ids = []
        for node_id, text_embedding in zip(node_ids, text_embeddings):
            normalized_node_type = self._normalize_node_type(node_types.get(node_id, "other"))
            node_type_id = self.node_type_to_index.get(normalized_node_type, self.node_type_to_index.get("other", 0))
            node_type_ids.append(node_type_id)
            type_one_hot = F.one_hot(torch.tensor(node_type_id), num_classes=self.n_node_types).to(dtype=torch.float32)
            degree = float(degree_lookup.get(node_id, 0))
            normalized_degree = degree / max(max_degree, 1)
            log_degree = math.log1p(degree) / max(math.log1p(max_degree), 1.0)
            degree_features = torch.tensor([normalized_degree, log_degree], dtype=torch.float32)
            node_feature_rows.append(torch.cat((text_embedding, type_one_hot, degree_features), dim=0))

        node_features = torch.stack(node_feature_rows, dim=0)
        edge_index, edge_features = self._build_edge_tensors(bundle=bundle, node_ids=node_ids)
        return GraphInputs(
            node_ids=node_ids,
            node_names=[node_names[node_id] for node_id in node_ids],
            node_type_ids=torch.tensor(node_type_ids, dtype=torch.long),
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features,
            node_scores=node_scores,
        )

    def _build_edge_tensors(self, bundle: EvidenceBundle, node_ids: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if not bundle.subgraph_edges:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, DEFAULT_EDGE_EMBED_DIM + 2), dtype=torch.float32),
            )

        node_index_lookup = {node_id: index for index, node_id in enumerate(node_ids)}
        edge_pairs = []
        edge_feature_rows = []
        for edge in bundle.subgraph_edges:
            if edge.head not in node_index_lookup or edge.tail not in node_index_lookup:
                continue
            edge_pairs.append((node_index_lookup[edge.head], node_index_lookup[edge.tail]))
            relation_index = self.relation_to_index.get(edge.relation, 0)
            relation_index = min(relation_index, self.n_relations - 1)
            relation_embedding = self.edge_emb(torch.tensor(relation_index, dtype=torch.long, device=self.device)).detach().cpu()
            reliability_features = torch.tensor([edge.source_reliability, edge.amg_confidence], dtype=torch.float32)
            edge_feature_rows.append(torch.cat((relation_embedding, reliability_features), dim=0))

        if not edge_pairs:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, DEFAULT_EDGE_EMBED_DIM + 2), dtype=torch.float32),
            )

        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_inputs = torch.stack(edge_feature_rows, dim=0).to(dtype=torch.float32)
        return edge_index, edge_inputs

    def _encode_passages(
        self,
        bundle: EvidenceBundle,
        node_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, list[str], dict[str, float]]:
        if not bundle.pubmed_passages:
            return torch.zeros((0, self.hidden_dim), dtype=torch.float32, device=self.device), [], {}

        passage_ids = [f"{DEFAULT_PASSAGE_PREFIX}{passage.pmid}" for passage in bundle.pubmed_passages]
        passage_texts = [self._passage_text(passage.title, passage.abstract) for passage in bundle.pubmed_passages]
        raw_passage_embeddings = self.passage_text_encoder.encode(passage_texts, device=self.device)
        passage_hidden = self.passage_proj(raw_passage_embeddings)
        if node_hidden.size(0) > 0:
            attended_passages, _ = self.bridge_attn(
                query=passage_hidden.unsqueeze(0),
                key=node_hidden.unsqueeze(0),
                value=node_hidden.unsqueeze(0),
                need_weights=False,
            )
            passage_hidden = passage_hidden + attended_passages.squeeze(0)

        raw_scores = torch.tensor([passage.relevance_score for passage in bundle.pubmed_passages], dtype=torch.float32)
        normalized_scores = self._normalize_scores(raw_scores)
        passage_scores = {passage_id: float(score) for passage_id, score in zip(passage_ids, normalized_scores.tolist())}
        return passage_hidden, passage_ids, passage_scores

    def _rank_and_select(
        self,
        node_ids: Sequence[str],
        node_embeddings: torch.Tensor,
        node_scores: dict[str, float],
        passage_ids: Sequence[str],
        passage_embeddings: torch.Tensor,
        passage_scores: dict[str, float],
    ) -> tuple[torch.Tensor, list[str]]:
        ranked_items: list[tuple[float, str, torch.Tensor]] = []
        for index, node_id in enumerate(node_ids):
            ranked_items.append((float(node_scores.get(node_id, 0.0)), node_id, node_embeddings[index]))
        for index, passage_id in enumerate(passage_ids):
            ranked_items.append((float(passage_scores.get(passage_id, 0.0)), passage_id, passage_embeddings[index]))

        ranked_items.sort(key=lambda item: (-item[0], item[1]))
        selected_items = ranked_items[: self.max_len]
        if not selected_items:
            return torch.zeros((0, self.hidden_dim), dtype=torch.float32, device=self.device), []

        selected_embeddings = torch.stack([item[2] for item in selected_items], dim=0).to(self.device)
        selected_mapping = [item[1] for item in selected_items]
        return selected_embeddings, selected_mapping

    def _quantize_selection(
        self,
        selected_embeddings: torch.Tensor,
        selected_mapping: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, str]]:
        token_ids = torch.full((self.max_len,), self.padding_token_id, dtype=torch.long, device=self.device)
        codebook_indices = torch.full((self.max_len,), self.padding_token_id, dtype=torch.long, device=self.device)
        node_mapping = {index: DEFAULT_PAD_IDENTIFIER for index in range(self.max_len)}

        if selected_embeddings.size(0) == 0:
            return token_ids, codebook_indices, node_mapping

        quantized, indices, _ = self.vq(selected_embeddings)
        real_count = min(indices.size(0), self.max_len)
        token_ids[:real_count] = indices[:real_count]
        codebook_indices[:real_count] = indices[:real_count]
        for index, identifier in enumerate(selected_mapping[:real_count]):
            node_mapping[index] = identifier
        return token_ids, codebook_indices, node_mapping

    def _extract_pretraining_embeddings(self, sample: Any) -> torch.Tensor | None:
        if isinstance(sample, EvidenceBundle):
            graph_inputs = self._build_graph_inputs(sample)
            if not graph_inputs.node_ids:
                return None
            node_hidden = self.node_proj(graph_inputs.node_features.to(self.device))
            node_type_ids = graph_inputs.node_type_ids.to(self.device)
            edge_index = graph_inputs.edge_index.to(self.device)
            raw_edge_features = graph_inputs.edge_features.to(self.device)
            edge_features = self.edge_proj(raw_edge_features) if raw_edge_features.numel() > 0 else raw_edge_features
            for layer in self.hgt:
                node_hidden = layer(node_hidden, edge_index, edge_features, node_type_ids)
            return node_hidden

        if isinstance(sample, torch.Tensor):
            if sample.ndim == 1:
                sample = sample.unsqueeze(0)
            return sample.to(dtype=torch.float32)

        if isinstance(sample, dict):
            for key in ("node_features", "x", "embeddings"):
                value = sample.get(key)
                if isinstance(value, torch.Tensor):
                    return value.to(dtype=torch.float32)

        tensor_candidate = getattr(sample, "x", None)
        if isinstance(tensor_candidate, torch.Tensor):
            return tensor_candidate.to(dtype=torch.float32)
        return None

    def _compute_personalized_pagerank(
        self,
        graph: nx.DiGraph,
        node_ids: Sequence[str],
        seed_node_ids: Sequence[str],
    ) -> dict[str, float]:
        if not graph.nodes:
            return {}
        if not graph.edges:
            return {node_id: (1.0 if node_id in seed_node_ids else 0.0) for node_id in node_ids}

        personalization = {node_id: 0.0 for node_id in graph.nodes}
        active_seeds = [node_id for node_id in seed_node_ids if node_id in personalization]
        if not active_seeds:
            active_seeds = list(node_ids)
        restart_probability = 1.0 / max(len(active_seeds), 1)
        for node_id in active_seeds:
            personalization[node_id] = restart_probability
        return nx.pagerank(graph, alpha=DEFAULT_PPR_ALPHA, personalization=personalization)

    def _normalize_node_type(self, node_type: str) -> str:
        normalized = node_type.strip().lower()
        if normalized in self.node_type_to_index:
            return normalized
        if normalized == "other":
            return normalized
        return ID_PREFIX_TO_NODE_TYPE.get(normalized, "other")

    def _infer_node_type(self, identifier: str) -> str:
        if ":" in identifier:
            prefix = identifier.split(":", 1)[0].strip().lower()
            return self._normalize_node_type(prefix)
        return "other"

    @staticmethod
    def _humanize_identifier(identifier: str) -> str:
        normalized = identifier.split(":", 1)[-1]
        return normalized.replace("_", " ").replace("-", " ").strip() or identifier

    @staticmethod
    def _slugify(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-") or "entity"

    @staticmethod
    def _normalize_scores(values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values
        min_value = values.min()
        max_value = values.max()
        if torch.isclose(min_value, max_value):
            zero = torch.zeros((), dtype=values.dtype, device=values.device)
            if torch.isclose(max_value, zero):
                return torch.zeros_like(values)
            return torch.ones_like(values)
        return (values - min_value) / (max_value - min_value)

    @staticmethod
    def _passage_text(title: str, abstract: str) -> str:
        return " ".join(part for part in [title, abstract] if part).strip()


def _resolve_model_artifact(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    candidate = Path(model_name)
    if candidate.is_absolute() and candidate.exists():
        return str(candidate)
    if str(model_name).replace("\\", "/").startswith("data/"):
        resolved = KaggleEnv.path(model_name)
        if resolved.exists():
            return str(resolved)
    return model_name


def get_sapbert_path(explicit_path: str | Path | None = None) -> str | None:
    resolver_candidate = KagglePathResolver.resolve_sapbert()
    if resolver_candidate is not None:
        logger.info("SapBERT found via resolver: %s", resolver_candidate)
        return resolver_candidate

    if explicit_path is not None:
        raw = Path(explicit_path)
        resolved = raw if raw.is_absolute() else KaggleEnv.path(raw)
        if resolved.exists():
            logger.info("SapBERT found at explicit path: %s", resolved)
            return str(resolved)
        logger.warning("SapBERT not found at explicit path: %s", explicit_path)

    fallback_candidates = (DEFAULT_SAPBERT_KAGGLE_PATH, KaggleEnv.path(DEFAULT_SAPBERT_LOCAL_PATH), DEFAULT_SAPBERT_LOCAL_PATH)
    for candidate in fallback_candidates:
        if candidate.exists():
            logger.info("SapBERT found at fallback candidate: %s", candidate)
            return str(candidate)

    logger.warning("SapBERT not found in known locations; fallback encoder will be used.")
    return None


__all__ = ["MedicalGraphEmbedder", "get_sapbert_path"]
