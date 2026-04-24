"""Pseudocode evaluation suite and report."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EvaluationReport:
    graph_version: str
    kg: dict
    entity_linking: dict
    retrieval: dict
    reasoning: dict
    trm: dict


class EvaluationSuite:
    def __init__(self, kg_quality_evaluator, entity_linking_evaluator, retrieval_evaluator, reasoning_evaluator, trm_evaluator, report_writer):
        self.kg_quality_evaluator = kg_quality_evaluator
        self.entity_linking_evaluator = entity_linking_evaluator
        self.retrieval_evaluator = retrieval_evaluator
        self.reasoning_evaluator = reasoning_evaluator
        self.trm_evaluator = trm_evaluator
        self.report_writer = report_writer

    def run_all(self, graph_version: str) -> EvaluationReport:
        report = EvaluationReport(graph_version, self.kg_quality_evaluator.evaluate(graph_version), self.entity_linking_evaluator.evaluate(graph_version), self.retrieval_evaluator.evaluate(graph_version), self.reasoning_evaluator.evaluate(graph_version), self.trm_evaluator.evaluate_latest(graph_version))
        self.report_writer.write(report)
        return report
