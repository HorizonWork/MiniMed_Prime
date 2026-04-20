from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


PostHocAuditMode = Literal["answer_only", "answer_plus_cot"]
PostHocClaimType = Literal[
    "factual",
    "logical",
    "contextual",
    "sourceattribution",
    "lifecritical",
    "temporal",
    "population",
]
PostHocVerdict = Literal["supported", "unsupported", "contradicted", "out_of_scope"]
PostHocRecommendation = Literal["ACCEPT", "REVISE", "ABSTAIN"]
PostHocSampleLabel = Literal["golden", "partial", "reject"]


class PostHocSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PostHocAuditClaim(PostHocSchemaModel):
    text: str
    type: PostHocClaimType
    verdict: PostHocVerdict
    evidence_id: str
    severity: int = Field(ge=0, le=3)
    rationale: str


class PostHocReasoningGap(PostHocSchemaModel):
    step: str
    severity: int = Field(ge=0, le=3)


class PostHocAuditOutput(PostHocSchemaModel):
    claims: list[PostHocAuditClaim]
    reasoning_gaps: list[PostHocReasoningGap] = Field(default_factory=list)
    unsupported_evidence_references: list[str] = Field(default_factory=list)
    faithfulness_score: float = Field(ge=0.0, le=1.0)
    h5_present: bool
    overall_recommendation: PostHocRecommendation
    abstention_reason: str | None = None
    sample_label: PostHocSampleLabel
    audit_metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _apply_recommendation_and_label_policy(self) -> PostHocAuditOutput:
        has_severity_three = any(claim.severity == 3 for claim in self.claims) or any(
            gap.severity == 3 for gap in self.reasoning_gaps
        )
        if self.h5_present or has_severity_three:
            self.overall_recommendation = "ABSTAIN"
            self.sample_label = "reject"
            if self.abstention_reason is None:
                self.abstention_reason = "Life-critical or severity-3 issue detected."
            return self

        if self.overall_recommendation == "ABSTAIN":
            self.sample_label = "reject"
            if self.abstention_reason is None:
                self.abstention_reason = "Auditor requested abstention."
            return self

        has_unsupported = any(claim.verdict in {"unsupported", "contradicted"} for claim in self.claims)
        has_major_gap = any(gap.severity >= 2 for gap in self.reasoning_gaps)
        if self.overall_recommendation == "REVISE" or has_unsupported or has_major_gap:
            self.overall_recommendation = "REVISE"
            self.sample_label = "partial"
            self.abstention_reason = None
            return self

        self.abstention_reason = None
        if self.faithfulness_score >= 0.8 and not self.reasoning_gaps:
            self.overall_recommendation = "ACCEPT"
            self.sample_label = "golden"
        else:
            self.overall_recommendation = "REVISE"
            self.sample_label = "partial"
        return self

