from __future__ import annotations

from pathlib import Path

from src.layers.layer1_retrieval import AgenticRetriever, EntityLinker, PrimeKGExtractor, PubMedRetriever
from src.schemas import PubMedPassage, QuestionEntity
from src.utils.pubmed_client import PubMedArticle, PubMedClient


class FakePubMedClient(PubMedClient):
    def __init__(self, cache_path: Path) -> None:
        super().__init__(cache_path=cache_path)

    def search_and_fetch(self, query: str, retmax: int | None = None, use_cache: bool = True) -> list[PubMedArticle]:
        return [
            PubMedArticle(
                pmid="12345678",
                title="Warfarin and ibuprofen",
                abstract="Ibuprofen increases bleeding risk when combined with warfarin.",
            ),
            PubMedArticle(
                pmid="23456789",
                title="Atrial fibrillation management",
                abstract="Warfarin prevents embolic stroke in atrial fibrillation.",
            ),
        ]


def write_sample_primekg_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "x_id,x_name,x_type,y_id,y_name,y_type,relation,display_relation,source,supporting_pmids",
                "drug:warfarin,warfarin,drug,disease:af,atrial fibrillation,disease,indication,indication,drugbank,23456789",
                "drug:warfarin,warfarin,drug,drug:ibuprofen,ibuprofen,drug,drug_drug,drug interaction,drugbank,12345678",
                "drug:ibuprofen,ibuprofen,drug,disease:bleeding,bleeding,disease,contraindication,contraindication,drugcentral,12345678",
                "drug:warfarin,warfarin,drug,effect:bleeding_risk,bleeding risk,effect,drug_effect,drug effect,drugcentral,12345678",
                "disease:af,atrial fibrillation,disease,disease:stroke,stroke,disease,disease_disease,associated disease,primekg,23456789",
            ]
        ),
        encoding="utf-8",
    )


def write_protein_only_primekg_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "x_id,x_name,x_type,y_id,y_name,y_type,relation,display_relation,source,supporting_pmids",
                "protein:ace2,ACE2,protein,protein:tmprss2,TMPRSS2,protein,protein_protein,ppi,primekg,12345678",
            ]
        ),
        encoding="utf-8",
    )


def test_entity_linker_fallback_extracts_medical_entities() -> None:
    linker = EntityLinker()

    entities = linker.link("Is ibuprofen contraindicated with warfarin in atrial fibrillation?")

    surfaces = {entity.surface.lower() for entity in entities}
    assert "ibuprofen" in surfaces
    assert "warfarin" in surfaces
    assert "atrial fibrillation" in surfaces
    assert linker.backend_used == "rule_based"


def test_entity_linker_classifies_missing_scispacy_package_as_expected_fallback() -> None:
    linker = EntityLinker()

    reason, expected = linker._classify_scispacy_exception(ModuleNotFoundError("No module named 'scispacy'"))

    assert reason == "scispacy_package_missing"
    assert expected is True


def test_entity_linker_classifies_missing_scispacy_model_as_expected_fallback() -> None:
    linker = EntityLinker()

    reason, expected = linker._classify_scispacy_exception(
        OSError("[E050] Can't find model 'en_core_sci_lg'. It doesn't seem to be a Python package or a valid path.")
    )

    assert reason == "scispacy_model_missing"
    assert expected is True


def test_entity_linker_classifies_unexpected_scispacy_errors_as_real_failures() -> None:
    linker = EntityLinker()

    reason, expected = linker._classify_scispacy_exception(RuntimeError("unexpected linker registry failure"))

    assert reason == "scispacy_initialization_failed"
    assert expected is False


def test_primekg_extractor_extracts_ranked_two_hop_edges(tmp_path: Path) -> None:
    primekg_path = tmp_path / "primekg_sample.csv"
    write_sample_primekg_csv(primekg_path)
    extractor = PrimeKGExtractor(primekg_path=primekg_path)

    seed_entities = [
        QuestionEntity(surface="warfarin", cui="C0043031", primekg_node_id=None, entity_type="drug"),
        QuestionEntity(surface="ibuprofen", cui="C0020740", primekg_node_id=None, entity_type="drug"),
    ]
    edges = extractor.extract_2hop(seed_entities=seed_entities, top_k=10, relation_filter={"drug_drug", "contraindication"})

    edge_ids = {edge.edge_id for edge in edges}
    relations = {edge.relation for edge in edges}
    assert edge_ids
    assert "drug_drug" in relations
    assert relations.issubset({"drug_drug", "contraindication"})


def test_primekg_extractor_compact_fallback_when_routed_filter_empty(tmp_path: Path) -> None:
    primekg_path = tmp_path / "primekg_sample.csv"
    write_sample_primekg_csv(primekg_path)
    extractor = PrimeKGExtractor(primekg_path=primekg_path)
    seed_entities = [
        QuestionEntity(surface="warfarin", cui="C0043031", primekg_node_id=None, entity_type="drug"),
    ]

    edges = extractor.extract_2hop(seed_entities=seed_entities, top_k=10, question_type="etiology", use_relation_filter=True)

    assert edges
    assert extractor.last_relation_filter_stats["fallback_stage"] == "compact_fallback"
    assert extractor.last_relation_filter_stats["edges_before"] >= extractor.last_relation_filter_stats["edges_after"]
    assert "disease_disease" in {edge.relation for edge in edges}


def test_primekg_extractor_falls_back_to_original_behavior_if_compact_still_empty(tmp_path: Path) -> None:
    primekg_path = tmp_path / "primekg_protein_only.csv"
    write_protein_only_primekg_csv(primekg_path)
    extractor = PrimeKGExtractor(primekg_path=primekg_path)
    seed_entities = [
        QuestionEntity(surface="ACE2", cui=None, primekg_node_id=None, entity_type="protein"),
    ]

    edges = extractor.extract_2hop(seed_entities=seed_entities, top_k=10, question_type="diagnosis", use_relation_filter=True)

    assert edges
    assert {edge.relation for edge in edges} == {"protein_protein"}
    assert extractor.last_relation_filter_stats["fallback_stage"] == "original_behavior"


def test_primekg_extractor_dosage_route_keeps_drug_effect(tmp_path: Path) -> None:
    primekg_path = tmp_path / "primekg_sample.csv"
    write_sample_primekg_csv(primekg_path)
    extractor = PrimeKGExtractor(primekg_path=primekg_path)
    seed_entities = [
        QuestionEntity(surface="warfarin", cui="C0043031", primekg_node_id=None, entity_type="drug"),
    ]

    edges = extractor.extract_2hop(seed_entities=seed_entities, top_k=10, question_type="dosage", use_relation_filter=True)

    assert "drug_effect" in {edge.relation for edge in edges}


def test_primekg_extractor_prefers_kg_csv_over_feature_tables(tmp_path: Path) -> None:
    (tmp_path / "disease_features.csv").write_text("node_id,feature\n1,0.5\n", encoding="utf-8")
    write_sample_primekg_csv(tmp_path / "kg.csv")

    extractor = PrimeKGExtractor(primekg_path=tmp_path)

    assert extractor.edge_lookup


def test_primekg_extractor_loads_edges_nodes_layout(tmp_path: Path) -> None:
    (tmp_path / "nodes.csv").write_text(
        "\n".join(
            [
                "node_index,node_id,node_type,node_name,node_source",
                "1,drug:warfarin,drug,warfarin,drugbank",
                "2,drug:ibuprofen,drug,ibuprofen,drugbank",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "edges.csv").write_text(
        "\n".join(
            [
                "relation,display_relation,x_index,y_index",
                "drug_drug,drug interaction,1,2",
            ]
        ),
        encoding="utf-8",
    )

    extractor = PrimeKGExtractor(primekg_path=tmp_path)

    assert list(extractor.edge_lookup.values())[0].head == "drug:warfarin"
    assert list(extractor.edge_lookup.values())[0].tail == "drug:ibuprofen"


def test_primekg_extractor_loads_optional_node_cui_from_edges_nodes_layout(tmp_path: Path) -> None:
    (tmp_path / "nodes.csv").write_text(
        "\n".join(
            [
                "node_index,node_id,node_type,node_name,node_source,node_cui",
                "1,disease:h_pylori,disease,Helicobacter pylori infectious disease,primekg,C0019163",
                "2,anatomy:colon,anatomy,Colon,primekg,C0009368",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "edges.csv").write_text(
        "\n".join(
            [
                "relation,display_relation,x_index,y_index",
                "disease_disease,associated disease,1,1",
            ]
        ),
        encoding="utf-8",
    )

    extractor = PrimeKGExtractor(primekg_path=tmp_path)

    assert extractor.node_cui_index["c0019163"] == {"disease:h_pylori"}


def test_primekg_extractor_resolves_alias_surface_against_generic_disease_suffix(tmp_path: Path) -> None:
    (tmp_path / "nodes.csv").write_text(
        "\n".join(
            [
                "node_index,node_id,node_type,node_name,node_source",
                "1,disease:h_pylori,disease,Helicobacter pylori infectious disease,primekg",
                "2,drug:warfarin,drug,warfarin,drugbank",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "edges.csv").write_text(
        "\n".join(
            [
                "relation,display_relation,x_index,y_index",
                "disease_disease,associated disease,1,1",
            ]
        ),
        encoding="utf-8",
    )

    extractor = PrimeKGExtractor(primekg_path=tmp_path)
    resolved = extractor.resolve_seed_entities(
        [QuestionEntity(surface="H pylori", cui=None, primekg_node_id=None, entity_type="disease")]
    )

    assert resolved[0].primekg_node_id == "disease:h_pylori"


def test_primekg_extractor_resolves_possessive_surface_to_matching_anatomy_node(tmp_path: Path) -> None:
    (tmp_path / "nodes.csv").write_text(
        "\n".join(
            [
                "node_index,node_id,node_type,node_name,node_source",
                "1,anatomy:colles_fascia,anatomy,Colles fascia,primekg",
                "2,anatomy:perineum,anatomy,Perineum,primekg",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "edges.csv").write_text(
        "\n".join(
            [
                "relation,display_relation,x_index,y_index",
                "anatomy_anatomy,parent-child,1,2",
            ]
        ),
        encoding="utf-8",
    )

    extractor = PrimeKGExtractor(primekg_path=tmp_path)
    resolved = extractor.resolve_seed_entities(
        [QuestionEntity(surface="Colle's fascia", cui=None, primekg_node_id=None, entity_type="anatomy")]
    )

    assert resolved[0].primekg_node_id == "anatomy:colles_fascia"


def test_primekg_extractor_matches_real_node_type_variants(tmp_path: Path) -> None:
    (tmp_path / "nodes.csv").write_text(
        "\n".join(
            [
                "node_index,node_id,node_type,node_name,node_source",
                "1,protein:phyhip,gene/protein,PHYHIP,NCBI",
                "2,symptom:bleeding,effect/phenotype,bleeding risk,primekg",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "edges.csv").write_text(
        "\n".join(
            [
                "relation,display_relation,x_index,y_index",
                "protein_protein,ppi,1,1",
                "phenotype_phenotype,associated phenotype,2,2",
            ]
        ),
        encoding="utf-8",
    )

    extractor = PrimeKGExtractor(primekg_path=tmp_path)
    resolved = extractor.resolve_seed_entities(
        [
            QuestionEntity(surface="PHYHIP", cui=None, primekg_node_id=None, entity_type="protein"),
            QuestionEntity(surface="bleeding risk", cui=None, primekg_node_id=None, entity_type="symptom"),
        ]
    )

    assert resolved[0].primekg_node_id == "protein:phyhip"
    assert resolved[1].primekg_node_id == "symptom:bleeding"


def test_pubmed_retriever_returns_pubmed_passages(tmp_path: Path) -> None:
    retriever = PubMedRetriever(cache_path=tmp_path / "cache.jsonl", client=FakePubMedClient(cache_path=tmp_path / "cache.jsonl"))
    retriever.config.query_encoder_model_name = None
    retriever.config.article_encoder_model_name = None
    retriever.config.cross_encoder_model_name = None

    passages = retriever.retrieve("warfarin ibuprofen interaction", k=2)

    assert len(passages) == 2
    assert all(isinstance(passage, PubMedPassage) for passage in passages)
    assert {passage.pmid for passage in passages} == {"12345678", "23456789"}


def test_agentic_retriever_builds_evidence_bundle(tmp_path: Path) -> None:
    primekg_path = tmp_path / "primekg_sample.csv"
    write_sample_primekg_csv(primekg_path)
    agentic_retriever = AgenticRetriever(
        primekg_path=primekg_path,
        pubmed_cache_path=tmp_path / "pubmed_cache.jsonl",
        kg_extractor=PrimeKGExtractor(primekg_path=primekg_path),
        pubmed=PubMedRetriever(
            cache_path=tmp_path / "pubmed_cache.jsonl",
            client=FakePubMedClient(cache_path=tmp_path / "pubmed_cache.jsonl"),
        ),
        linker=EntityLinker(),
    )
    agentic_retriever.pubmed.config.query_encoder_model_name = None
    agentic_retriever.pubmed.config.article_encoder_model_name = None
    agentic_retriever.pubmed.config.cross_encoder_model_name = None

    bundle = agentic_retriever.retrieve("Is ibuprofen contraindicated with warfarin in atrial fibrillation?")

    assert bundle.question_type == "drug_interaction"
    assert bundle.question_entities
    assert bundle.subgraph_edges
    assert bundle.pubmed_passages
    assert bundle.metadata["question_type"] == "drug_interaction"
    assert bundle.metadata["entity_linker_backend"] == "rule_based"
    assert bundle.metadata["pubmed_backend"] == "bm25_only"
    relation_filter_stats = bundle.metadata["relation_filter"]
    assert relation_filter_stats["question_type_used"] == "drug_interaction"
    assert "requested_relations" in relation_filter_stats
    assert "matched_relations" in relation_filter_stats
    assert "edges_before" in relation_filter_stats
    assert "edges_after" in relation_filter_stats
