"""Unit tests for the PrimeKG CSV parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from minimed_rag.ingestion.base import RawFile
from minimed_rag.ingestion.primekg.parser import (
    iter_primekg_edges,
    parse_primekg_edges,
)

CSV_HEADER = "x_id,x_name,x_type,relation,y_id,y_name,y_type,source\n"
CSV_ROWS = (
    "MONDO:1,diabetes,disease,disease_protein,NCBI:1,INS,gene/protein,curated\n"
    "DrugBank:1,metformin,drug,indication,MONDO:1,diabetes,disease,curated\n"
    "MONDO:1,diabetes,disease,disease_phenotype_positive,HP:1,polyuria,phenotype,curated\n"
)


def _write_csv(tmp_path: Path, body: str = CSV_HEADER + CSV_ROWS) -> Path:
    path = tmp_path / "kg.csv"
    path.write_text(body, encoding="utf-8")
    return path


def test_iter_primekg_edges_streams_all_rows(tmp_path: Path):
    path = _write_csv(tmp_path)
    edges = list(iter_primekg_edges(path))

    assert len(edges) == 3
    assert edges[0].source_node_id == "MONDO:1"
    assert edges[0].relation == "disease_protein"
    assert edges[1].source_node_type == "drug"
    assert edges[1].target_node_type == "disease"
    assert edges[2].target_node_name == "polyuria"
    assert edges[0].source == "curated"


def test_iter_primekg_edges_respects_limit(tmp_path: Path):
    path = _write_csv(tmp_path)
    edges = list(iter_primekg_edges(path, limit=2))
    assert len(edges) == 2


def test_iter_primekg_edges_missing_columns_raises(tmp_path: Path):
    # Drop `relation` column
    bad = "x_id,x_name,x_type,y_id,y_name,y_type,source\nMONDO:1,x,disease,NCBI:1,y,gene,curated\n"
    path = _write_csv(tmp_path, body=bad)
    with pytest.raises(ValueError, match="missing required columns"):
        list(iter_primekg_edges(path))


def test_iter_primekg_edges_empty_source_field(tmp_path: Path):
    body = CSV_HEADER + "MONDO:1,x,disease,disease_disease,MONDO:2,y,disease,\n"
    path = _write_csv(tmp_path, body=body)
    edges = list(iter_primekg_edges(path))
    assert edges[0].source is None


def test_parse_primekg_edges_legacy_bytes_interface():
    raw = RawFile(name="kg.csv", bytes=(CSV_HEADER + CSV_ROWS).encode("utf-8"))
    edges = parse_primekg_edges(raw)
    assert len(edges) == 3
    assert edges[1].relation == "indication"
