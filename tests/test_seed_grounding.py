from __future__ import annotations

import json
from pathlib import Path

from src.retrieval.seed_grounding import (
    DeterministicSeedGrounder,
    EntityGroundingJudge,
    LLMAssistedSeedGrounder,
    PrimeKGNodeCatalog,
)
from src.schemas import QuestionEntity


def _write_nodes_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "node_index,node_id,node_type,node_name,node_source",
                "1,disease:h_pylori,disease,Helicobacter pylori infectious disease,primekg",
                "2,anatomy:common_hepatic_artery,anatomy,common hepatic artery,primekg",
                "3,anatomy:right_gastroepiploic_artery,anatomy,right gastroepiploic artery,primekg",
                "4,anatomy:urogenital_diaphragm,anatomy,urogenital diaphragm,primekg",
                "5,drug:warfarin,drug,warfarin,drugbank",
                "6,protein:warfarin,protein,warfarin,primekg",
            ]
        ),
        encoding="utf-8",
    )


class StubGroundingJudge(EntityGroundingJudge):
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        super().__init__(model_name="stub-grounding", device="cpu")
        self.backend = "stub"

    def _load_backend(self) -> None:
        self.backend = "stub"

    def _generate_structured_text(self, prompt: str) -> str:
        del prompt
        return self.outputs.pop(0)


def test_deterministic_grounder_links_known_alias_and_typo_cases(tmp_path: Path) -> None:
    _write_nodes_csv(tmp_path / "nodes.csv")
    catalog = PrimeKGNodeCatalog(nodes_csv=tmp_path / "nodes.csv")
    grounder = DeterministicSeedGrounder(catalog)
    entities = [
        QuestionEntity(surface="H pylori", cui=None, primekg_node_id=None, entity_type="disease"),
        QuestionEntity(surface="common hepatic aery", cui=None, primekg_node_id=None, entity_type="anatomy"),
        QuestionEntity(surface="right gastroepiploic aery", cui=None, primekg_node_id=None, entity_type="anatomy"),
        QuestionEntity(surface="urogenital diaphragm", cui=None, primekg_node_id=None, entity_type="anatomy"),
    ]

    decisions = grounder.ground("test question", entities)

    assert [decision.linked_node_id for decision in decisions] == [
        "disease:h_pylori",
        "anatomy:common_hepatic_artery",
        "anatomy:right_gastroepiploic_artery",
        "anatomy:urogenital_diaphragm",
    ]


def test_deterministic_grounder_leaves_unsupported_surfaces_unresolved(tmp_path: Path) -> None:
    _write_nodes_csv(tmp_path / "nodes.csv")
    catalog = PrimeKGNodeCatalog(nodes_csv=tmp_path / "nodes.csv")
    grounder = DeterministicSeedGrounder(catalog)
    entities = [
        QuestionEntity(surface="Typhoid", cui=None, primekg_node_id=None, entity_type="disease"),
        QuestionEntity(surface="Colle's fascia", cui=None, primekg_node_id=None, entity_type="anatomy"),
    ]

    decisions = grounder.ground("test question", entities)

    assert [decision.linked_node_id for decision in decisions] == [None, None]


def test_deterministic_grounder_prefers_matching_node_type_when_names_are_close(tmp_path: Path) -> None:
    _write_nodes_csv(tmp_path / "nodes.csv")
    catalog = PrimeKGNodeCatalog(nodes_csv=tmp_path / "nodes.csv")
    grounder = DeterministicSeedGrounder(catalog)
    entity = QuestionEntity(surface="warfarin", cui=None, primekg_node_id=None, entity_type="drug")

    decision = grounder.ground("test question", [entity])[0]

    assert decision.linked_node_id == "drug:warfarin"
    assert decision.candidates[0].node_type == "drug"


def test_llm_assisted_grounder_rejects_invented_node_ids(tmp_path: Path) -> None:
    _write_nodes_csv(tmp_path / "nodes.csv")
    catalog = PrimeKGNodeCatalog(nodes_csv=tmp_path / "nodes.csv")
    grounder = LLMAssistedSeedGrounder(catalog, model_name="heuristic", device="cpu")
    grounder.judge = StubGroundingJudge(
        [
            json.dumps(
                {
                    "decisions": [
                        {
                            "surface": "H pylori",
                            "normalized_surface": "helicobacter pylori",
                            "entity_type": "disease",
                            "decision": "link",
                            "chosen_node_id": "node:invented",
                            "chosen_node_name": "Invented Node",
                            "confidence": 0.9,
                            "rationale": "bad choice",
                        }
                    ]
                }
            )
        ]
    )
    grounder.llm_used = True
    grounder.backend_used = "llm_assisted_v1_stub"
    entity = QuestionEntity(surface="H pylori", cui=None, primekg_node_id=None, entity_type="disease")

    decisions = grounder.ground("Which H pylori test is best?", [entity])

    assert decisions[0].linked_node_id == "disease:h_pylori"
    assert decisions[0].decision_reason == "llm_invalid_choice_fallback"
    assert grounder.fallback_reason == "llm_invalid_choice"


def test_llm_assisted_grounder_falls_back_to_deterministic_without_real_backend(tmp_path: Path) -> None:
    _write_nodes_csv(tmp_path / "nodes.csv")
    catalog = PrimeKGNodeCatalog(nodes_csv=tmp_path / "nodes.csv")
    grounder = LLMAssistedSeedGrounder(catalog, model_name="heuristic", device="cpu")
    entity = QuestionEntity(surface="H pylori", cui=None, primekg_node_id=None, entity_type="disease")

    decisions = grounder.ground("Which H pylori test is best?", [entity])

    assert grounder.llm_used is False
    assert grounder.backend_used == "deterministic_v2"
    assert grounder.fallback_reason == "llm_backend_unavailable"
    assert decisions[0].linked_node_id == "disease:h_pylori"
