"""Pseudocode RDF exporter."""
from __future__ import annotations


class RDFExporter:
    def export_assertions(self, assertions: list) -> str:
        triples = []
        for assertion in assertions:
            triples.append(f"<{assertion.subject_id}> <{assertion.predicate}> <{assertion.object_id}> .")
        return "\n".join(triples)
