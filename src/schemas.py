from __future__ import annotations

from typing import Any, Literal, TypeAlias
from uuid import uuid4

import torch
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

EntityType: TypeAlias = Literal["disease", "drug", "protein", "symptom", "anatomy", "other"]
QuestionType: TypeAlias = Literal["diagnosis", "drug_interaction", "dosage", "etiology", "factoid", "other"]
PrimeKGRelation: TypeAlias = Literal[
    "anatomy_anatomy",
    "anatomy_protein_absent",
    "anatomy_protein_present",
    "bioprocess_bioprocess",
    "bioprocess_protein",
    "cellcomp_cellcomp",
    "cellcomp_protein",
    "contraindication",
    "disease_disease",
    "disease_phenotype_negative",
    "disease_phenotype_positive",
    "disease_protein",
    "drug_drug",
    "drug_effect",
    "drug_protein",
    "exposure_bioprocess",
    "exposure_cellcomp",
    "exposure_disease",
    "exposure_exposure",
    "exposure_molfunc",
    "exposure_protein",
    "indication",
    "molfunc_molfunc",
    "molfunc_protein",
    "off-label use",
    "pathway_pathway",
    "pathway_protein",
    "phenotype_phenotype",
    "phenotype_protein",
    "protein_protein",
]
ClaimType: TypeAlias = Literal[
    "factual",
    "logical",
    "contextual",
    "source_attribution",
    "life_critical",
    "temporal",
    "population",
]
ClaimVerdictLabel: TypeAlias = Literal["supported", "unsupported", "contradicted", "out_of_scope"]
Recommendation: TypeAlias = Literal["ACCEPT", "REVISE", "ABSTAIN"]
TraceEntry: TypeAlias = dict[str, Any]

TRM_INPUT_SHAPE: tuple[int, int] = (1, 256)
TRM_PUZZLE_IDENTIFIERS_SHAPE: tuple[int] = (1,)
DEFAULT_ABSTENTION_REASON_BLOCKING_CLAIM = "Blocking severity-3 claim detected."
DEFAULT_ABSTENTION_REASON_H5 = "H5 blocking issue detected."
DEFAULT_ABSTENTION_REASON_GENERIC = "Judge recommended abstention."


def _coerce_int64_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(dtype=torch.long)
    if isinstance(value, (list, tuple)):
        return torch.tensor(value, dtype=torch.long)
    raise ValueError("Tensor fields must be provided as torch.Tensor or JSON-serializable lists.")


def _validate_tensor_shape(value: torch.Tensor, expected_shape: tuple[int, ...], field_name: str) -> torch.Tensor:
    actual_shape = tuple(value.shape)
    if actual_shape != expected_shape:
        raise ValueError(f"{field_name} must have shape {expected_shape}; received {actual_shape}.")
    return value


def _normalize_provenance_reference(value: Any, *, allow_none: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("Provenance references must be strings.")

    normalized = value.strip()
    if normalized != value or not normalized:
        raise ValueError("Provenance references must not contain surrounding whitespace.")

    if allow_none and normalized == "none":
        return normalized

    if normalized.startswith("edge_id:"):
        normalized = f"edge:{normalized[len('edge_id:'):]}"

    if normalized.startswith("edge:"):
        edge_id = normalized[len("edge:") :]
        if edge_id and edge_id.strip() == edge_id:
            return normalized
    if normalized.startswith("PMID:"):
        pmid = normalized[len("PMID:") :]
        if pmid and pmid.strip() == pmid:
            return normalized

    raise ValueError("Provenance references must use 'edge:<id>' or 'PMID:<id>'.")


class SchemaModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class QuestionEntity(SchemaModel):
    surface: str
    cui: str | None = None
    primekg_node_id: str | None = None
    entity_type: EntityType


class KGEdge(SchemaModel):
    edge_id: str
    head: str
    tail: str
    relation: PrimeKGRelation
    display_relation: str
    source_reliability: float = Field(ge=0.0, le=1.0)
    amg_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    supporting_pmids: list[str]


class PubMedPassage(SchemaModel):
    pmid: str
    title: str
    abstract: str
    relevance_score: float


class EvidenceBundle(SchemaModel):
    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question_text: str
    question_type: QuestionType
    question_entities: list[QuestionEntity]
    subgraph_edges: list[KGEdge]
    pubmed_passages: list[PubMedPassage]
    metadata: dict[str, Any] = Field(default_factory=dict)


class TRMInputBundle(SchemaModel):
    inputs: torch.Tensor
    puzzle_identifiers: torch.Tensor
    original_graph: EvidenceBundle

    @field_validator("inputs", "puzzle_identifiers", mode="before")
    @classmethod
    def _convert_to_tensor(cls, value: Any) -> torch.Tensor:
        return _coerce_int64_tensor(value)

    @field_validator("inputs")
    @classmethod
    def _validate_inputs_shape(cls, value: torch.Tensor) -> torch.Tensor:
        return _validate_tensor_shape(value, TRM_INPUT_SHAPE, "inputs")

    @field_validator("puzzle_identifiers")
    @classmethod
    def _validate_puzzle_identifiers_shape(cls, value: torch.Tensor) -> torch.Tensor:
        return _validate_tensor_shape(value, TRM_PUZZLE_IDENTIFIERS_SHAPE, "puzzle_identifiers")

    @field_serializer("inputs", "puzzle_identifiers", when_used="json")
    def _serialize_tensor(self, value: torch.Tensor) -> list[Any]:
        return value.detach().cpu().tolist()


class Path(SchemaModel):
    nodes: list[str]
    edges: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_pmids: list[str]

    @model_validator(mode="after")
    def _validate_path_structure(self) -> Path:
        expected_edge_count = max(len(self.nodes) - 1, 0)
        if len(self.edges) != expected_edge_count:
            raise ValueError(
                f"Path edges must contain exactly one edge per node hop; expected {expected_edge_count}, "
                f"received {len(self.edges)}."
            )
        return self


class TRMOutput(SchemaModel):
    ranked_paths: list[Path]
    validity_score: float = Field(ge=0.0, le=1.0)
    contradiction_flag: bool
    trace: list[TraceEntry]

    @field_validator("ranked_paths")
    @classmethod
    def _validate_ranked_paths(cls, value: list[Path]) -> list[Path]:
        if len(value) > 5:
            raise ValueError("ranked_paths must contain at most 5 entries.")
        return value


class ClaimVerdict(SchemaModel):
    text: str
    claim_type: ClaimType
    verdict: ClaimVerdictLabel
    evidence_id: str
    severity: int
    rationale: str

    @field_validator("evidence_id", mode="before")
    @classmethod
    def _normalize_evidence_id(cls, value: Any) -> str:
        return _normalize_provenance_reference(value, allow_none=True)

    @field_validator("severity")
    @classmethod
    def _validate_severity(cls, value: int) -> int:
        if value < 0 or value > 3:
            raise ValueError("severity must be between 0 and 3.")
        return value


class JudgeOutput(SchemaModel):
    claims: list[ClaimVerdict]
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    h5_present: bool
    overall_recommendation: Recommendation
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def _apply_abstention_policy(self) -> JudgeOutput:
        has_blocking_claim = any(claim.severity == 3 for claim in self.claims)
        if self.h5_present:
            self.overall_recommendation = "ABSTAIN"
        elif has_blocking_claim:
            self.overall_recommendation = "ABSTAIN"

        if self.overall_recommendation == "ABSTAIN":
            if self.abstention_reason is None:
                if self.h5_present:
                    self.abstention_reason = DEFAULT_ABSTENTION_REASON_H5
                elif has_blocking_claim:
                    self.abstention_reason = DEFAULT_ABSTENTION_REASON_BLOCKING_CLAIM
                else:
                    self.abstention_reason = DEFAULT_ABSTENTION_REASON_GENERIC
        else:
            self.abstention_reason = None

        return self


class AnswerWithTrace(SchemaModel):
    answer_text: str
    question_id: str
    provenance: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    abstention: bool
    abstention_reason: str | None = None
    judge_outputs: list[JudgeOutput]
    trm_trace: list[TraceEntry] | None = None

    @field_validator("provenance")
    @classmethod
    def _normalize_provenance(cls, value: list[str]) -> list[str]:
        return [_normalize_provenance_reference(item) for item in value]


__all__ = [
    "AnswerWithTrace",
    "ClaimType",
    "ClaimVerdict",
    "ClaimVerdictLabel",
    "DEFAULT_ABSTENTION_REASON_BLOCKING_CLAIM",
    "DEFAULT_ABSTENTION_REASON_GENERIC",
    "DEFAULT_ABSTENTION_REASON_H5",
    "EntityType",
    "EvidenceBundle",
    "JudgeOutput",
    "KGEdge",
    "Path",
    "PrimeKGRelation",
    "PubMedPassage",
    "QuestionEntity",
    "QuestionType",
    "Recommendation",
    "TRMInputBundle",
    "TRMOutput",
    "TRM_INPUT_SHAPE",
    "TRM_PUZZLE_IDENTIFIERS_SHAPE",
    "TraceEntry",
]
