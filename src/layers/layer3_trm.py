from __future__ import annotations

import logging
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from src.models.trm_wrapper import TRMModelBundle, load_trm_model_bundle, move_nested_tensors_to_device, stablemax_cross_entropy
from src.schemas import EvidenceBundle, KGEdge, Path as ReasoningPath, TRMOutput
from src.utils.checkpoint import CheckpointManager, resume_or_initialize
from src.utils.kaggle_env import KaggleEnv, T4Hardening
from src.utils.structured_logger import DEFAULT_LOG_DIR, StructuredLogger

try:
    from loguru import logger
except ImportError:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())


DEFAULT_TOP_K_PATHS = 5
DEFAULT_BEAM_WIDTH = 5
DEFAULT_MAX_PATH_LENGTH = 4
DEFAULT_NLI_MODEL_NAME = "roberta-large-mnli"
DEFAULT_NEGATIVE_RELATIONS = {"contraindication", "disease_phenotype_negative", "anatomy_protein_absent"}
DEFAULT_NEGATION_CUES = (
    "no benefit",
    "no evidence",
    "not recommended",
    "not effective",
    "did not",
    "does not",
    "fails to",
    "contraindicated",
    "avoid",
    "worsens",
    "increases risk",
)
DEFAULT_POSITIVE_CUES = (
    "recommended",
    "effective",
    "prevents",
    "reduces",
    "treats",
    "improves",
    "supports",
    "beneficial",
    "indicated",
)
DEFAULT_PADDING_IDENTIFIER = "__pad__"
DEFAULT_PUBMED_PREFIX = "PMID:"
DEFAULT_PATH_STABILITY_EPSILON = 1e-3
DEFAULT_PATH_STABILITY_STEPS = 2
DEFAULT_HALT_LOSS_WEIGHT = 0.1


@dataclass(slots=True)
class CandidateNode:
    node_id: str
    score: float
    position: int
    token_id: int


@dataclass(slots=True)
class DecodedPathCandidate:
    nodes: list[str]
    edges: list[str]
    supporting_pmids: set[str] = field(default_factory=set)
    edge_confidences: list[float] = field(default_factory=list)
    ranking_score: float = 0.0


class TRMReasoner:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_steps: int = 16,
        conf_threshold: float = 0.85,
        contradiction_threshold: float = 0.5,
        allow_official_untrained: bool = False,
        strict_validate: bool = True,
    ) -> None:
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.max_steps = max_steps
        self.conf_threshold = conf_threshold
        self.contradiction_threshold = contradiction_threshold
        self.strict_validate = strict_validate
        self.model_path = model_path
        self.event_logger = StructuredLogger("layer3_trm", DEFAULT_LOG_DIR)
        T4Hardening.verify_fp16("float16")
        resolved_model_path = Path(model_path)
        if not resolved_model_path.is_absolute():
            resolved_model_path = KaggleEnv.path(resolved_model_path)
        self.model_bundle = load_trm_model_bundle(
            model_path=resolved_model_path,
            device=self.device,
            allow_official_untrained=allow_official_untrained,
            strict_validate=strict_validate,
        )
        self.model = self.model_bundle.model
        self.trm_is_real = self.model_bundle.source == "official" and self.model_bundle.checkpoint_loaded
        self.trm_is_medical_compatible = bool(self.model_bundle.medical_compatible)
        if self.trm_is_real:
            self.backend_used = "samsung_trm"
        elif self.model_bundle.source == "official_untrained":
            self.backend_used = "samsung_trm_untrained"
        else:
            self.backend_used = "fallback_trm"
        self._needs_vocab_remap = self.trm_is_real and not self.trm_is_medical_compatible
        self.model.eval()
        self.optimizer: torch.optim.Optimizer | None = None
        self.grad_scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        self.nli_pipeline = self._load_local_nli_pipeline()
        self.event_logger.log_event(
            "trm_init",
            {
                "is_real_checkpoint": self.trm_is_real,
                "checkpoint_path": str(self.model_bundle.checkpoint_path) if self.model_bundle.checkpoint_path is not None else "",
                "checkpoint_loaded": self.model_bundle.checkpoint_loaded,
                "medical_compatible": self.model_bundle.medical_compatible,
                "load_diagnostics": list(self.model_bundle.load_diagnostics),
                "backend_used": self.backend_used,
                "input_remap_active": self._needs_vocab_remap,
                "strict_validate": self.strict_validate,
                "latency_ms": 0.0,
            },
        )
        if hasattr(self.model, "gradient_checkpointing_enable"):
            try:
                self.model.gradient_checkpointing_enable()  # type: ignore[call-arg]
            except Exception:
                logger.debug("TRM model does not support gradient checkpointing enablement.")

    def reason(self, embedder_output: dict[str, Any]) -> TRMOutput:
        inputs = self._coerce_long_tensor(embedder_output["inputs"], expected_rank=2)
        puzzle_identifiers = self._coerce_long_tensor(embedder_output["puzzle_identifiers"], expected_rank=1)
        node_mapping = dict(embedder_output.get("node_mapping", {}))
        evidence_bundle = self._extract_evidence_bundle(embedder_output)
        primekg_edges = evidence_bundle.subgraph_edges if evidence_bundle is not None else embedder_output.get("primekg_edges", [])

        batch = {
            "inputs": inputs.to(self.device),
            "puzzle_identifiers": puzzle_identifiers.to(self.device),
        }
        batch, remap_info = self._align_batch_to_model_vocab(batch)
        batch["labels"] = batch["inputs"]
        with torch.device(self.device):
            carry = self.model.initial_carry(batch)
        carry = move_nested_tensors_to_device(carry, self.device)

        trace: list[dict[str, Any]] = []
        final_outputs: dict[str, torch.Tensor] | None = None
        final_token_confidence = 0.0
        final_halt_probability = 0.0
        previous_logits: torch.Tensor | None = None
        previous_answer_state: torch.Tensor | None = None
        stable_step_count = 0
        halt_reason = "max_steps"
        trace.append(
            {
                "stage": "trm_backend",
                "backend_used": self.backend_used,
                "trm_is_real": self.trm_is_real,
                "checkpoint_path": str(self.model_bundle.checkpoint_path) if self.model_bundle.checkpoint_path is not None else None,
                "checkpoint_loaded": self.model_bundle.checkpoint_loaded,
                "medical_compatible": self.model_bundle.medical_compatible,
                "load_diagnostics": list(self.model_bundle.load_diagnostics),
                "repo_root": str(self.model_bundle.repo_root) if self.model_bundle.repo_root is not None else None,
                "input_remap": remap_info if remap_info else None,
            }
        )
        self.event_logger.log_event(
            "recursion_start",
            {
                "max_steps": self.max_steps,
                "T": getattr(self.model_bundle.config, "H_cycles", None),
                "n": getattr(self.model_bundle.config, "L_cycles", None),
                "backend_used": self.backend_used,
                "latency_ms": 0.0,
            },
        )

        autocast_context = (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad():
            for step_index in range(self.max_steps):
                with autocast_context:
                    carry, outputs = self.model(carry, batch)
                final_outputs = outputs

                logits = outputs["logits"].float()
                halt_probability = torch.sigmoid(outputs["q_halt_logits"].float()).mean().item()
                token_confidences = F.softmax(logits, dim=-1).max(dim=-1).values
                mean_token_confidence = token_confidences.mean().item()
                top_positions = torch.topk(token_confidences[0], k=min(5, token_confidences.shape[-1])).indices.tolist()
                answer_state = self._extract_answer_state(carry)
                if answer_state is not None and previous_answer_state is not None:
                    delta_norm = self._relative_delta_norm(answer_state, previous_answer_state)
                    path_stable = delta_norm < DEFAULT_PATH_STABILITY_EPSILON
                    stable_step_count = stable_step_count + 1 if path_stable else 0
                    previous_answer_state = answer_state.detach().clone()
                elif answer_state is not None:
                    delta_norm = 0.0
                    path_stable = False
                    previous_answer_state = answer_state.detach().clone()
                elif previous_logits is None:
                    delta_norm = 0.0
                    path_stable = False
                else:
                    delta_norm = self._relative_delta_norm(logits, previous_logits)
                    path_stable = delta_norm < DEFAULT_PATH_STABILITY_EPSILON
                    stable_step_count = stable_step_count + 1 if path_stable else 0
                previous_logits = logits.clone()

                trace.append(
                    {
                        "step": step_index + 1,
                        "halt_probability": halt_probability,
                        "halt_prob": halt_probability,
                        "mean_token_confidence": mean_token_confidence,
                        "confidence": mean_token_confidence,
                        "halted": bool(carry.halted.all().item()),
                        "top_positions": top_positions,
                        "path_stable": path_stable,
                        "stable_step_count": stable_step_count,
                        "delta_norm": delta_norm,
                    }
                )
                self.event_logger.log_event(
                    "recursion_step",
                    {
                        "step": step_index + 1,
                        "halt_prob": halt_probability,
                        "path_confidence": mean_token_confidence,
                        "delta_norm": delta_norm,
                        "contradiction_score": 0.0,
                        "backend_used": self.backend_used,
                        "latency_ms": 0.0,
                    },
                )
                final_token_confidence = mean_token_confidence
                final_halt_probability = halt_probability

                if carry.halted.all():
                    halt_reason = "halt_max_steps"
                    break
                if stable_step_count >= DEFAULT_PATH_STABILITY_STEPS:
                    halt_reason = "path_stability"
                    break
                if halt_probability >= self.conf_threshold:
                    halt_reason = "confidence_threshold"
                    break

        if final_outputs is None:
            raise RuntimeError("TRM reasoning produced no outputs.")

        logits = final_outputs["logits"].float()
        ranked_paths = self.decode_paths(
            y_hat_logits=logits,
            node_mapping=node_mapping,
            primekg_edges=list(primekg_edges),
            top_k=DEFAULT_TOP_K_PATHS,
        )
        self.event_logger.log_event(
            "paths_decoded",
            {
                "top_k": DEFAULT_TOP_K_PATHS,
                "path_lengths": [len(path.edges) for path in ranked_paths],
                "path_confidences": [path.confidence for path in ranked_paths],
                "backend_used": self.backend_used,
                "latency_ms": 0.0,
            },
        )

        top_node_ids = [node_id for path in ranked_paths for node_id in path.nodes]
        contradiction_score = self.detect_contradiction(top_node_ids, evidence_bundle)
        contradiction_flag = contradiction_score >= self.contradiction_threshold
        max_path_confidence = max((path.confidence for path in ranked_paths), default=0.0)
        validity_score = max(0.0, min(1.0, max_path_confidence))
        trace.append(
            {
                "final_halt_probability": final_halt_probability,
                "final_token_confidence": final_token_confidence,
                "contradiction_score": contradiction_score,
                "validity_score": validity_score,
                "validity_score_source": "best_path_multiplicative_confidence",
                "steps_taken": len(trace),
            }
        )
        self.event_logger.log_event(
            "recursion_halted",
            {
                "halt_reason": halt_reason,
                "steps_taken": len(trace) - 1,
                "final_confidence": validity_score,
                "backend_used": self.backend_used,
                "latency_ms": 0.0,
            },
        )

        return TRMOutput(
            ranked_paths=ranked_paths[:DEFAULT_TOP_K_PATHS],
            validity_score=validity_score,
            contradiction_flag=contradiction_flag,
            trace=trace,
        )

    @staticmethod
    def _extract_answer_state(carry: Any) -> torch.Tensor | None:
        inner_carry = getattr(carry, "inner_carry", None)
        answer_state = getattr(inner_carry, "y_state", None)
        if isinstance(answer_state, torch.Tensor):
            return answer_state.float()
        answer_state = getattr(inner_carry, "y", None)
        if isinstance(answer_state, torch.Tensor):
            return answer_state.float()
        return None

    @staticmethod
    def _relative_delta_norm(current: torch.Tensor, previous: torch.Tensor) -> float:
        numerator = torch.linalg.norm(current.float() - previous.float())
        denominator = torch.linalg.norm(current.float()).clamp_min(1e-8)
        return float((numerator / denominator).detach().cpu().item())

    def _align_batch_to_model_vocab(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if not self._needs_vocab_remap:
            return batch, {}

        prepared = dict(batch)
        remap_info: dict[str, Any] = {}

        inputs = prepared["inputs"]
        vocab_size = max(int(getattr(self.model_bundle.config, "vocab_size", 0)), 1)
        remapped_inputs = torch.remainder(inputs, vocab_size)
        remapped_input_count = int((remapped_inputs != inputs).sum().item())
        if remapped_input_count:
            prepared["inputs"] = remapped_inputs
            remap_info["input_vocab_size"] = vocab_size
            remap_info["input_tokens_remapped"] = remapped_input_count

        puzzle_identifiers = prepared["puzzle_identifiers"]
        puzzle_limit = max(int(getattr(self.model_bundle.config, "num_puzzle_identifiers", 0)), 1)
        remapped_puzzle_ids = puzzle_identifiers.clamp(min=0, max=puzzle_limit - 1)
        remapped_puzzle_count = int((remapped_puzzle_ids != puzzle_identifiers).sum().item())
        if remapped_puzzle_count:
            prepared["puzzle_identifiers"] = remapped_puzzle_ids
            remap_info["puzzle_identifier_limit"] = puzzle_limit
            remap_info["puzzle_identifiers_remapped"] = remapped_puzzle_count

        if remap_info:
            self.event_logger.log_event(
                "input_vocab_remap",
                {
                    "backend_used": self.backend_used,
                    "trm_is_real": self.trm_is_real,
                    "medical_compatible": self.trm_is_medical_compatible,
                    "details": dict(remap_info),
                    "latency_ms": 0.0,
                },
            )
        return prepared, remap_info

    def decode_paths(
        self,
        y_hat_logits: torch.Tensor,
        node_mapping: dict[int, str],
        primekg_edges: list[KGEdge],
        top_k: int = 5,
    ) -> list[ReasoningPath]:
        candidate_nodes = self._extract_candidate_nodes(y_hat_logits, node_mapping)
        if not candidate_nodes:
            return []

        candidate_score_lookup = {candidate.node_id: candidate.score for candidate in candidate_nodes}
        adjacency: dict[str, list[KGEdge]] = {}
        for edge in primekg_edges:
            adjacency.setdefault(edge.head, []).append(edge)

        ranked_candidates: list[tuple[float, ReasoningPath]] = []
        beam: list[DecodedPathCandidate] = []
        for candidate in candidate_nodes[:DEFAULT_BEAM_WIDTH]:
            beam.append(
                DecodedPathCandidate(
                    nodes=[candidate.node_id],
                    edges=[],
                    ranking_score=candidate.score,
                )
            )

        for initial_candidate in beam:
            ranked_candidates.append(
                (
                    initial_candidate.ranking_score,
                    ReasoningPath(
                        nodes=list(initial_candidate.nodes),
                        edges=[],
                        confidence=initial_candidate.ranking_score,
                        supporting_pmids=[],
                    ),
                )
            )

        for _ in range(DEFAULT_MAX_PATH_LENGTH - 1):
            new_beam: list[DecodedPathCandidate] = []
            for partial_path in beam:
                tail_node = partial_path.nodes[-1]
                for edge in adjacency.get(tail_node, []):
                    if edge.tail in partial_path.nodes or edge.tail not in candidate_score_lookup:
                        continue
                    edge_weight = edge.source_reliability * edge.amg_confidence
                    edge_confidences = [*partial_path.edge_confidences, edge_weight]
                    path_confidence = self.compute_multiplicative_confidence(edge_confidences)
                    ranking_score = path_confidence * candidate_score_lookup[edge.tail]
                    new_candidate = DecodedPathCandidate(
                        nodes=[*partial_path.nodes, edge.tail],
                        edges=[*partial_path.edges, edge.edge_id],
                        supporting_pmids=partial_path.supporting_pmids | set(edge.supporting_pmids),
                        edge_confidences=edge_confidences,
                        ranking_score=ranking_score,
                    )
                    new_beam.append(new_candidate)
                    ranked_candidates.append(
                        (
                            ranking_score,
                            ReasoningPath(
                                nodes=list(new_candidate.nodes),
                                edges=list(new_candidate.edges),
                                confidence=path_confidence,
                                supporting_pmids=sorted(new_candidate.supporting_pmids),
                            ),
                        )
                    )

            if not new_beam:
                break
            new_beam.sort(key=lambda candidate: (-candidate.ranking_score, candidate.nodes))
            beam = new_beam[:DEFAULT_BEAM_WIDTH]

        deduplicated: dict[tuple[tuple[str, ...], tuple[str, ...]], tuple[float, ReasoningPath]] = {}
        for ranking_score, path in ranked_candidates:
            path_key = (tuple(path.nodes), tuple(path.edges))
            if path_key not in deduplicated or ranking_score > deduplicated[path_key][0]:
                deduplicated[path_key] = (ranking_score, path)

        connected_paths = [
            item for item in deduplicated.values() if item[1].edges
        ]
        singleton_paths = [
            item for item in deduplicated.values() if not item[1].edges
        ]
        connected_paths.sort(
            key=lambda item: (-len(item[1].edges), -item[0], -item[1].confidence, item[1].nodes),
        )
        singleton_paths.sort(
            key=lambda item: (-item[0], -item[1].confidence, item[1].nodes),
        )
        ranked_paths = connected_paths + singleton_paths
        return [path for _, path in ranked_paths[:top_k]]

    def detect_contradiction(self, top_tokens: Sequence[Any], evidence_bundle: EvidenceBundle | None) -> float:
        if evidence_bundle is None or not evidence_bundle.subgraph_edges or not evidence_bundle.pubmed_passages:
            return 0.0

        candidate_nodes = {token for token in top_tokens if isinstance(token, str) and not token.startswith(DEFAULT_PUBMED_PREFIX)}
        contradiction_score = 0.0

        for edge in evidence_bundle.subgraph_edges:
            if candidate_nodes and edge.head not in candidate_nodes and edge.tail not in candidate_nodes:
                continue
            edge_negative = edge.relation in DEFAULT_NEGATIVE_RELATIONS or "contra" in edge.display_relation.lower()
            head_text = self._humanize_identifier(edge.head).lower()
            tail_text = self._humanize_identifier(edge.tail).lower()
            for passage in evidence_bundle.pubmed_passages:
                passage_text = f"{passage.title} {passage.abstract}".lower()
                if head_text not in passage_text and tail_text not in passage_text:
                    continue

                heuristic_score = self._heuristic_contradiction_score(
                    passage_text=passage_text,
                    edge_negative=edge_negative,
                )
                contradiction_score = max(contradiction_score, heuristic_score)

                if self.nli_pipeline is not None:
                    nli_score = self._nli_contradiction_score(
                        premise=passage.abstract,
                        hypothesis=self._edge_to_hypothesis(edge),
                    )
                    contradiction_score = max(contradiction_score, nli_score)

        edge_polarities: dict[tuple[str, str], set[bool]] = {}
        for edge in evidence_bundle.subgraph_edges:
            pair_key = (edge.head, edge.tail)
            edge_polarities.setdefault(pair_key, set()).add(edge.relation in DEFAULT_NEGATIVE_RELATIONS or "contra" in edge.display_relation.lower())
        if any(len(polarities) > 1 for polarities in edge_polarities.values()):
            contradiction_score = max(contradiction_score, 0.6)

        return contradiction_score

    def train_step(
        self,
        batch: dict[str, Any],
        *,
        step: int | None = None,
        checkpoint_dir: Path | None = None,
        save_every: int = 2500,
    ) -> dict[str, float]:
        started_at = time.perf_counter()
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        inputs = self._coerce_long_tensor(batch.get("trm_input", batch.get("inputs")), expected_rank=2).to(self.device)
        labels = self._coerce_long_tensor(batch.get("gold_path_tokens", batch.get("labels")), expected_rank=2).to(self.device)
        puzzle_identifiers = self._coerce_long_tensor(batch.get("puzzle_ids", batch.get("puzzle_identifiers")), expected_rank=1).to(self.device)
        train_batch = {
            "inputs": inputs,
            "labels": labels,
            "puzzle_identifiers": puzzle_identifiers,
        }
        train_batch, _ = self._align_batch_to_model_vocab(
            {
                "inputs": train_batch["inputs"],
                "puzzle_identifiers": train_batch["puzzle_identifiers"],
            }
        )
        train_batch["labels"] = train_batch["inputs"]

        self.model.train()
        with torch.device(self.device):
            carry = self.model.initial_carry(train_batch)
        carry = move_nested_tensors_to_device(carry, self.device)
        final_outputs: dict[str, torch.Tensor] | None = None
        steps_taken = 0
        autocast_context = (
            torch.cuda.amp.autocast(dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )

        self.optimizer.zero_grad(set_to_none=True)
        with autocast_context:
            for _ in range(self.max_steps):
                carry, outputs = self.model(carry, train_batch)
                final_outputs = outputs
                steps_taken += 1
                if carry.halted.all():
                    break

            if final_outputs is None:
                raise RuntimeError("TRM training produced no outputs.")

            logits = final_outputs["logits"]
            label_targets = train_batch["labels"]
            token_loss = stablemax_cross_entropy(logits, label_targets)
            predictions = logits.argmax(dim=-1)
            valid_mask = label_targets != -100
            if torch.any(valid_mask):
                seq_is_correct = ((predictions == label_targets) | ~valid_mask).all(dim=-1)
            else:
                seq_is_correct = torch.ones((label_targets.shape[0],), dtype=torch.bool, device=label_targets.device)
            q_halt_loss = F.binary_cross_entropy_with_logits(
                final_outputs["q_halt_logits"],
                seq_is_correct.to(dtype=final_outputs["q_halt_logits"].dtype),
            )
            total_loss = token_loss + (DEFAULT_HALT_LOSS_WEIGHT * q_halt_loss)

        if self.device.type == "cuda":
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()

        self.model.eval()
        valid_token_count = int(valid_mask.sum().detach().cpu().item())
        if valid_token_count:
            token_accuracy = float((predictions[valid_mask] == label_targets[valid_mask]).float().mean().detach().cpu().item())
        else:
            token_accuracy = 1.0
        sequence_accuracy = float(seq_is_correct.float().mean().detach().cpu().item())
        learning_rate = float(self.optimizer.param_groups[0].get("lr", 0.0)) if self.optimizer.param_groups else 0.0
        metrics = {
            "loss": float(total_loss.detach().cpu()),
            "token_loss": float(token_loss.detach().cpu()),
            "q_halt_loss": float(q_halt_loss.detach().cpu()),
            "token_accuracy": token_accuracy,
            "sequence_accuracy": sequence_accuracy,
            "valid_token_count": float(valid_token_count),
            "steps_taken": float(steps_taken),
            "learning_rate": learning_rate,
        }
        current_step = int(step or 0)
        self.event_logger.log_metric("trm_train_loss", metrics["loss"], step=current_step)
        self.event_logger.log_metric("trm_token_accuracy", metrics["token_accuracy"], step=current_step)
        self.event_logger.log_event(
            "trm_train_step",
            {
                "step": current_step,
                "loss": metrics["loss"],
                "token_loss": metrics["token_loss"],
                "q_halt_loss": metrics["q_halt_loss"],
                "token_accuracy": token_accuracy,
                "sequence_accuracy": sequence_accuracy,
                "valid_token_count": valid_token_count,
                "batch_size": int(inputs.shape[0]),
                "seq_len": int(inputs.shape[1]),
                "steps_taken": steps_taken,
                "learning_rate": learning_rate,
                "backend_used": self.backend_used,
                "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            },
        )
        if checkpoint_dir is not None and current_step % max(save_every, 1) == 0:
            checkpoint_manager = CheckpointManager(checkpoint_dir)
            checkpoint_manager.save(
                current_step,
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict() if self.optimizer is not None else {},
                metrics=metrics,
            )
        return metrics

    def resume_training(self, checkpoint_dir: Path | None = None) -> int:
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        resolved_checkpoint_dir = checkpoint_dir or KaggleEnv.path("data/checkpoints/active")
        return resume_or_initialize(self.model, self.optimizer, resolved_checkpoint_dir)

    @staticmethod
    def compute_multiplicative_confidence(confidences: Sequence[float]) -> float:
        if not confidences:
            return 0.0
        product = 1.0
        for confidence in confidences:
            product *= float(confidence)
        if len(confidences) > 2:
            product *= float(confidences[0]) * float(confidences[-1])
        return float(product)

    def _extract_candidate_nodes(self, y_hat_logits: torch.Tensor, node_mapping: dict[int, str]) -> list[CandidateNode]:
        if y_hat_logits.ndim != 3:
            raise ValueError("y_hat_logits must have shape [batch, seq_len, vocab_size].")
        probabilities = F.softmax(y_hat_logits[0].float(), dim=-1)
        top_scores, top_token_ids = probabilities.max(dim=-1)

        candidates_by_node: dict[str, CandidateNode] = {}
        for position, (score, token_id) in enumerate(zip(top_scores.tolist(), top_token_ids.tolist())):
            node_id = node_mapping.get(position)
            if node_id is None or node_id == DEFAULT_PADDING_IDENTIFIER or node_id.startswith(DEFAULT_PUBMED_PREFIX):
                continue
            candidate = CandidateNode(node_id=node_id, score=float(score), position=position, token_id=int(token_id))
            if node_id not in candidates_by_node or candidate.score > candidates_by_node[node_id].score:
                candidates_by_node[node_id] = candidate

        candidates = sorted(
            candidates_by_node.values(),
            key=lambda candidate: (-candidate.score, candidate.position, candidate.node_id),
        )
        return candidates[: max(DEFAULT_BEAM_WIDTH * 2, DEFAULT_TOP_K_PATHS)]

    def _extract_evidence_bundle(self, embedder_output: dict[str, Any]) -> EvidenceBundle | None:
        for key in ("original_graph", "evidence_bundle", "bundle"):
            value = embedder_output.get(key)
            if isinstance(value, EvidenceBundle):
                return value
        return None

    def _heuristic_contradiction_score(self, passage_text: str, edge_negative: bool) -> float:
        has_negative_cue = any(cue in passage_text for cue in DEFAULT_NEGATION_CUES)
        has_positive_cue = any(cue in passage_text for cue in DEFAULT_POSITIVE_CUES)
        if edge_negative and has_positive_cue:
            return 0.8
        if (not edge_negative) and has_negative_cue:
            return 0.8
        if edge_negative == has_negative_cue:
            return 0.0
        return 0.35

    def _nli_contradiction_score(self, premise: str, hypothesis: str) -> float:
        if self.nli_pipeline is None:
            return 0.0
        try:
            output = self.nli_pipeline({"text": premise, "text_pair": hypothesis}, truncation=True)
        except Exception:
            return 0.0
        if isinstance(output, list):
            output = output[0]
        label = str(output.get("label", "")).lower()
        score = float(output.get("score", 0.0))
        if "contrad" in label:
            return score
        return 0.0

    def _load_local_nli_pipeline(self) -> Any | None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_NLI_MODEL_NAME, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_NLI_MODEL_NAME, local_files_only=True)
            return pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device.type == "cuda" else -1,
            )
        except Exception:
            return None

    @staticmethod
    def _coerce_long_tensor(value: Any, expected_rank: int) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.clone().detach().to(dtype=torch.long)
        else:
            tensor = torch.tensor(value, dtype=torch.long)
        if tensor.ndim != expected_rank:
            raise ValueError(f"Expected tensor rank {expected_rank}, received {tensor.ndim}.")
        return tensor

    @staticmethod
    def _edge_to_hypothesis(edge: KGEdge) -> str:
        return f"{TRMReasoner._humanize_identifier(edge.head)} {edge.display_relation} {TRMReasoner._humanize_identifier(edge.tail)}."

    @staticmethod
    def _humanize_identifier(identifier: str) -> str:
        normalized = identifier.split(":", 1)[-1]
        return normalized.replace("_", " ").replace("-", " ").strip() or identifier


__all__ = ["TRMReasoner"]
