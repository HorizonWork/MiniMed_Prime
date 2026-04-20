from __future__ import annotations

import json
from pathlib import Path

import torch

from src.layers.layer1_retrieval import AgenticRetriever, PrimeKGExtractor, PubMedRetriever
from src.schemas import QuestionEntity
from src.utils.pubmed_client import PubMedArticle, PubMedClient
from tools.trace_one_sample import TraceConfig, trace_one_sample


class StaticLinker:
    backend_used = "rule_based"

    def link(self, question: str) -> list[QuestionEntity]:
        lowered = question.lower()
        if "ibuprofen" in lowered or "warfarin" in lowered:
            return [
                QuestionEntity(surface="warfarin", cui="C0043031", primekg_node_id=None, entity_type="drug"),
                QuestionEntity(surface="ibuprofen", cui="C0020740", primekg_node_id=None, entity_type="drug"),
            ]
        if "metformin" in lowered:
            return [
                QuestionEntity(surface="metformin", cui="C0025598", primekg_node_id=None, entity_type="drug"),
                QuestionEntity(surface="kidney", cui="C0022646", primekg_node_id=None, entity_type="anatomy"),
            ]
        if "pneumonia" in lowered:
            return [
                QuestionEntity(surface="pneumonia", cui="C0032285", primekg_node_id=None, entity_type="disease"),
                QuestionEntity(surface="fever", cui="C0015967", primekg_node_id=None, entity_type="symptom"),
            ]
        return []


class FakePubMedClient(PubMedClient):
    def __init__(self, cache_path: Path) -> None:
        super().__init__(cache_path=cache_path)

    def search_and_fetch(self, query: str, retmax: int | None = None, use_cache: bool = True) -> list[PubMedArticle]:
        del query, retmax, use_cache
        return [
            PubMedArticle(
                pmid="12345678",
                title="Warfarin and ibuprofen",
                abstract="Ibuprofen increases bleeding risk when combined with warfarin.",
            ),
            PubMedArticle(
                pmid="23456789",
                title="Clinical diagnosis",
                abstract="Fever can support a pneumonia diagnosis in the right context.",
            ),
        ]


class FakeTraceEmbedder:
    padding_token_id = 4095
    codebook_size = 4096
    backend_used = "fake_hash"
    last_backend_used = {
        "node_text_encoder": "fake",
        "passage_text_encoder": "fake",
        "graph_encoder": "fake_hgt",
        "vector_quantizer": "fake_vq",
        "layer2_backend": "fake_hash",
    }

    def __call__(self, evidence):
        inputs = torch.full((1, 256), self.padding_token_id, dtype=torch.long)
        node_mapping = {index: "__pad__" for index in range(256)}
        nodes = sorted({node for edge in evidence.subgraph_edges for node in (edge.head, edge.tail)})
        for index, node_id in enumerate(nodes[:240]):
            inputs[0, index] = 100 + index
            node_mapping[index] = node_id
        return {
            "inputs": inputs,
            "puzzle_identifiers": torch.tensor([0], dtype=torch.long),
            "node_mapping": node_mapping,
            "backend_used": dict(self.last_backend_used),
        }


def test_trace_one_sample_runs_three_question_types_and_writes_artifacts(tmp_path: Path) -> None:
    source = _write_medreason_records(tmp_path)
    retriever = _make_retriever(tmp_path)
    output_dir = tmp_path / "traces"

    traces = [
        trace_one_sample(
            TraceConfig(source=source, split=None, sample_index=index, output_dir=output_dir, edge_mapper_backend="direct"),
            retriever=retriever,
            embedder=FakeTraceEmbedder(),
        )
        for index in range(3)
    ]

    assert [trace["sample_id"] for trace in traces] == ["drug-int", "dose", "diagnosis"]
    assert [trace["question_type"]["predicted"] for trace in traces] == [
        "drug_interaction",
        "dosage",
        "diagnosis",
    ]
    assert all(trace["final_trace_verdict"]["status"] == "partial" for trace in traces)
    assert all(trace["tensorization_preview"]["labeled_token_count"] > 0 for trace in traces)
    assert all(Path(trace["artifacts"]["trace_json"]).exists() for trace in traces)
    assert all(Path(trace["artifacts"]["summary_md"]).exists() for trace in traces)
    assert all(Path(trace["artifacts"]["subgraph_json"]).exists() for trace in traces)
    assert all(Path(trace["artifacts"]["pubmed_json"]).exists() for trace in traces)
    assert "bm25_score" in traces[0]["pubmed_retrieval"]["top_pmids"][0]


def test_trace_one_sample_rejects_missing_gold_edges_with_debug_summary(tmp_path: Path) -> None:
    source = tmp_path / "medreason_missing.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "missing-gold",
                "question": "Does ibuprofen interact with warfarin?",
                "answer": "Yes",
                "reasoning": "The reasoning cites [edge:E_MISSING].",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    trace = trace_one_sample(
        TraceConfig(source=source, split=None, output_dir=tmp_path / "traces", edge_mapper_backend="direct"),
        retriever=_make_retriever(tmp_path),
        embedder=FakeTraceEmbedder(),
    )

    assert trace["final_trace_verdict"]["status"] == "reject"
    assert trace["final_trace_verdict"]["failed_stage"] == "gold_mapping"
    assert trace["gold_mapping"]["edge_coverage"] == 0.0
    assert trace["gold_mapping"]["missing_gold_edge_ids"] == ["E_MISSING"]
    assert trace["tensorization_preview"]["labeled_token_count"] == 0

    summary = Path(trace["artifacts"]["summary_md"]).read_text(encoding="utf-8")
    assert "CRITICAL: labeled_token_count=0" in summary
    assert "E_MISSING" in summary


def test_trace_one_sample_supports_question_id_and_random_selection(tmp_path: Path) -> None:
    source = _write_medreason_records(tmp_path)
    retriever = _make_retriever(tmp_path)

    by_id = trace_one_sample(
        TraceConfig(source=source, split=None, question_id="dose", output_dir=tmp_path / "by_id", edge_mapper_backend="direct"),
        retriever=retriever,
        embedder=FakeTraceEmbedder(),
    )
    random_trace = trace_one_sample(
        TraceConfig(source=source, split=None, random_one=True, seed=7, output_dir=tmp_path / "random", edge_mapper_backend="direct"),
        retriever=retriever,
        embedder=FakeTraceEmbedder(),
    )

    assert by_id["record_index"] == 1
    assert by_id["sample_id"] == "dose"
    assert random_trace["sample_id"] in {"drug-int", "dose", "diagnosis"}


def test_trace_one_sample_can_embed_posthoc_audit(tmp_path: Path) -> None:
    source = _write_medreason_records(tmp_path)
    retriever = _make_retriever(tmp_path)

    trace = trace_one_sample(
        TraceConfig(
            source=source,
            split=None,
            sample_index=0,
            output_dir=tmp_path / "trace_with_audit",
            edge_mapper_backend="direct",
            run_posthoc_audit=True,
            posthoc_audit_mode="answer_only",
            posthoc_audit_model="heuristic",
            posthoc_audit_device="cpu",
        ),
        retriever=retriever,
        embedder=FakeTraceEmbedder(),
    )

    assert trace["posthoc_audit"]["status"] in {"ok", "warn"}
    assert trace["posthoc_audit"]["result"]["overall_recommendation"] in {"ACCEPT", "REVISE", "ABSTAIN"}
    assert trace["final_trace_verdict"]["posthoc_recommendation"] in {"ACCEPT", "REVISE", "ABSTAIN"}


def _make_retriever(tmp_path: Path) -> AgenticRetriever:
    primekg_path = tmp_path / "primekg.csv"
    _write_primekg(primekg_path)
    pubmed = PubMedRetriever(
        cache_path=tmp_path / "pubmed_cache.jsonl",
        client=FakePubMedClient(cache_path=tmp_path / "pubmed_cache.jsonl"),
    )
    pubmed.config.query_encoder_model_name = None
    pubmed.config.article_encoder_model_name = None
    pubmed.config.cross_encoder_model_name = None
    return AgenticRetriever(
        primekg_path=primekg_path,
        pubmed_cache_path=tmp_path / "pubmed_cache.jsonl",
        kg_extractor=PrimeKGExtractor(primekg_path=primekg_path),
        pubmed=pubmed,
        linker=StaticLinker(),
    )


def _write_medreason_records(tmp_path: Path) -> Path:
    source = tmp_path / "medreason.jsonl"
    rows = [
        {
            "id": "drug-int",
            "question": "Does ibuprofen interact with warfarin?",
            "answer": "Yes",
            "reasoning": "Supported by [edge:E_DRUG].",
        },
        {
            "id": "dose",
            "question": "What dose adjustment is needed for metformin in kidney disease?",
            "answer": "Reduce dose",
            "reasoning": "Supported by [edge:E_DOSE].",
        },
        {
            "id": "diagnosis",
            "question": "Most likely diagnosis for fever with pneumonia findings?",
            "answer": "Pneumonia",
            "reasoning": "Supported by [edge:E_DX].",
        },
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return source


def _write_primekg(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "edge_id,x_id,x_name,x_type,y_id,y_name,y_type,relation,display_relation,source,supporting_pmids",
                "E_DRUG,drug:warfarin,warfarin,drug,drug:ibuprofen,ibuprofen,drug,drug_drug,drug interaction,drugbank,12345678",
                "E_DOSE,drug:metformin,metformin,drug,anatomy:kidney,kidney,anatomy,drug_effect,dose effect,drugbank,12345678",
                "E_DX,disease:pneumonia,pneumonia,disease,symptom:fever,fever,phenotype,disease_phenotype_positive,has phenotype,primekg,23456789",
            ]
        ),
        encoding="utf-8",
    )
