from __future__ import annotations

import json
from pathlib import Path

from src.eval import (
    Evaluator,
    bootstrap_ci,
    contradiction_detection_rate,
    exact_match,
    f1_hallucination,
    macro_f1,
    mcnemar_test,
    ragas_faithfulness,
    reasoning_completeness_score,
    reasoning_necessity_score,
)
from src.orchestrator import MedicalReasoningSystemV3
from src.schemas import AnswerWithTrace, JudgeOutput


def test_exact_match_ignores_case_whitespace_and_provenance() -> None:
    assert exact_match("Direct answer: Yes [PMID:12345].", "yes") == 1.0


def test_macro_f1_handles_yes_no_maybe_labels() -> None:
    score = macro_f1(
        predictions=["yes", "no", "maybe", "yes"],
        golds=["yes", "no", "maybe", "no"],
    )

    assert score == 0.7777777777777777


def test_ragas_faithfulness_supports_abbreviation_expansion_and_numeric_tolerance() -> None:
    answer = "AF is treated with warfarin 5.1 mg/day [PMID:12345678]."
    context = ["Atrial fibrillation (AF) is treated with warfarin 5 mg/day in selected patients."]

    score = ragas_faithfulness(answer=answer, context=context, nli_model=None)

    assert score == 1.0


def test_f1_hallucination_from_claim_sets() -> None:
    precision, recall, f1 = f1_hallucination(
        claims_supported=["warfarin treats atrial fibrillation"],
        total_claims=["warfarin treats atrial fibrillation", "ibuprofen cures pneumonia"],
        gold_claims=["warfarin treats atrial fibrillation", "warfarin reduces stroke risk"],
    )

    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


def test_contradiction_detection_rate_flags_opposite_claims() -> None:
    rate = contradiction_detection_rate(
        claims=[
            "Warfarin is indicated for atrial fibrillation.",
            "Warfarin is not indicated for atrial fibrillation.",
        ],
        nli_model=None,
    )

    assert rate == 1.0


def test_reasoning_scores_measure_overlap() -> None:
    completeness = reasoning_completeness_score(
        predicted_edges=["edge:E1", "edge:E2"],
        gold_edges=["E1", "E2", "E3"],
    )
    necessity = reasoning_necessity_score(
        predicted_edges=["edge:E1", "edge:E2"],
        gold_edges=["E1", "E3"],
    )

    assert completeness == 2 / 3
    assert necessity == 0.5


def test_mcnemar_test_detects_three_percent_accuracy_difference() -> None:
    model_a_correct = [True] * 830 + [False] * 170
    model_b_correct = ([True] * 830) + ([True] * 30) + ([False] * 140)

    chi_square, p_value = mcnemar_test(model_a_correct=model_a_correct, model_b_correct=model_b_correct)

    assert chi_square > 0.0
    assert p_value < 0.05


def test_bootstrap_ci_returns_interval_covering_sample_mean() -> None:
    differences = [0.01, 0.03, 0.02, 0.04, 0.05, 0.02, 0.03]

    lower, upper = bootstrap_ci(differences, n_bootstrap=500)

    sample_mean = sum(differences) / len(differences)
    assert lower < upper
    assert lower <= sample_mean <= upper


class StubPipeline(MedicalReasoningSystemV3):
    def __init__(self, outputs: dict[str, AnswerWithTrace]) -> None:
        self.outputs = outputs

    def answer(self, question: str) -> AnswerWithTrace:
        return self.outputs[question]


def test_evaluator_runs_local_jsonl_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "pubmedqa.jsonl"
    records = [
        {
            "question": "Is warfarin indicated for atrial fibrillation?",
            "final_decision": "yes",
            "context": ["Warfarin is indicated for atrial fibrillation."],
            "gold_edges": ["E2"],
            "gold_claims": ["warfarin is indicated for atrial fibrillation"],
        },
        {
            "question": "Does ibuprofen cure pneumonia?",
            "final_decision": "no",
            "context": ["Ibuprofen is not indicated for pneumonia."],
            "gold_edges": ["E1"],
            "gold_claims": ["ibuprofen is not indicated for pneumonia"],
        },
    ]
    with dataset_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    empty_judge = JudgeOutput.model_validate(
        {
            "claims": [],
            "faithfulness_score": 1.0,
            "h5_present": False,
            "overall_recommendation": "ACCEPT",
            "abstention_reason": None,
        }
    )
    pipeline = StubPipeline(
        outputs={
            "Is warfarin indicated for atrial fibrillation?": AnswerWithTrace(
                answer_text="Yes [edge:E2].\nReasoning: Warfarin is indicated for atrial fibrillation [edge:E2] [PMID:23456789].\nConfidence level: high (0.90) [edge:E2].",
                question_id="q1",
                provenance=["edge:E2", "PMID:23456789"],
                confidence=0.9,
                abstention=False,
                abstention_reason=None,
                judge_outputs=[empty_judge],
                trm_trace=[],
            ),
            "Does ibuprofen cure pneumonia?": AnswerWithTrace(
                answer_text="No [edge:E1].\nReasoning: Ibuprofen does not cure pneumonia [edge:E1] [PMID:12345678].\nConfidence level: high (0.88) [edge:E1].",
                question_id="q2",
                provenance=["edge:E1", "PMID:12345678"],
                confidence=0.88,
                abstention=False,
                abstention_reason=None,
                judge_outputs=[empty_judge],
                trm_trace=[],
            ),
        }
    )

    evaluator = Evaluator(pipeline=pipeline, dataset=str(dataset_path))
    results = evaluator.run()

    assert results["samples"] == 2.0
    assert results["exact_match"] == 1.0
    assert results["macro_f1"] == 1.0
    assert results["abstention_rate"] == 0.0
    assert results["latency_p50_ms"] >= 0.0
    assert results["latency_p95_ms"] >= results["latency_p50_ms"]
    assert results["mean_provenance_count"] == 2.0
