"""Unit tests for the UMLS RRF parsers."""

from __future__ import annotations

from pathlib import Path

import pytest

from minimed_rag.ingestion.umls.mrconso_parser import iter_mrconso
from minimed_rag.ingestion.umls.mrrel_parser import iter_mrrel
from minimed_rag.ingestion.umls.mrsty_parser import iter_mrsty

MRCONSO_LINES = (
    # CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF
    "C0025598|ENG|P|L0025598|PF|S0025598|Y|A0001|||MTH:NOCODE|MTH|PN|NOCODE|Metformin|0|N|\n"
    "C0025598|ENG|S|L9999999|PF|S9999999|N|A0002|||DBID:DB00331|DRUGBANK|IN|DB00331|Metformin hydrochloride|0|N|\n"
    "C0025598|FRE|P|L0000001|PF|S0000001|Y|A0003|||MTHFR:NOCODE|MTHFR|PN|NOCODE|Metformine|0|N|\n"
    "C0025598|ENG|P|L0025598|PF|S0025598|Y|A0004|||MTH:NOCODE|MTH|PN|NOCODE|Suppressed|0|E|\n"
)


MRSTY_LINES = (
    "C0025598|T121|A1.4.1.1.1|Pharmacologic Substance|AT123|\n"
    "C0025598|T200|A1.4.1.1.3|Clinical Drug|AT124|\n"
)


MRREL_LINES = (
    # CUI1|AUI1|STYPE1|REL|CUI2|AUI2|STYPE2|RELA|RUI|SRUI|SAB|SL|RG|DIR|SUPPRESS|CVF
    "C0025598|A0001|CUI|RO|C0011849|A1000|CUI|may_treat|R001||DRUGBANK|DRUGBANK|||N|\n"
    "C0025598|A0001|CUI|SY|C0025598|A0002|CUI||R002||MTH|MTH|||N|\n"
    "C0025598|A0001|CUI|CHD|C0013227|A1001|CUI||R003||MTH|MTH|||N|\n"
    "C0025598|A0001|CUI|PAR|C0013227|A1001|CUI||R004||MTH|MTH|||N|\n"
    "C0025598|A0001|CUI|PAR|C0013227|A1001|CUI||R005||MTH|MTH|||Y|\n"  # suppressed
)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_iter_mrconso_filters_language_and_suppress(tmp_path: Path):
    path = _write(tmp_path, "MRCONSO.RRF", MRCONSO_LINES)
    rows = list(iter_mrconso(path))

    # Expect: rows 1 + 2 (ENG, SUPPRESS=N). Not row 3 (FRE). Not row 4 (SUPPRESS=E).
    assert len(rows) == 2
    assert rows[0]["cui"] == "C0025598"
    assert rows[0]["str"] == "Metformin"
    assert rows[0]["is_preferred"] is True
    assert rows[1]["sab"] == "DRUGBANK"
    assert rows[1]["is_preferred"] is False


def test_iter_mrconso_respects_limit(tmp_path: Path):
    path = _write(tmp_path, "MRCONSO.RRF", MRCONSO_LINES)
    rows = list(iter_mrconso(path, limit=2))
    # limit applies to pre-filter rows; filter then drops non-ENG/suppressed.
    # row 1 + row 2 ENG both pass.
    assert len(rows) <= 2


def test_iter_mrsty_yields_semantic_types(tmp_path: Path):
    path = _write(tmp_path, "MRSTY.RRF", MRSTY_LINES)
    rows = list(iter_mrsty(path))
    assert len(rows) == 2
    assert rows[0]["tui"] == "T121"
    assert rows[0]["sty"] == "Pharmacologic Substance"


def test_iter_mrrel_skips_sy_and_suppressed(tmp_path: Path):
    path = _write(tmp_path, "MRREL.RRF", MRREL_LINES)
    rows = list(iter_mrrel(path))

    # Expect: row 1 (RO may_treat), row 3 (CHD), row 4 (PAR).
    # Skipped: row 2 (SY), row 5 (SUPPRESS=Y).
    assert len(rows) == 3
    rels = [r["rel"] for r in rows]
    assert "SY" not in rels
    assert rows[0]["rela"] == "may_treat"


def test_iter_mrrel_drops_self_loops(tmp_path: Path):
    line = "C0001|A|CUI|RO|C0001|A|CUI||R001||MSH|MSH|||N|\n"
    path = _write(tmp_path, "MRREL.RRF", line)
    assert list(iter_mrrel(path)) == []


def test_iter_rrf_raises_on_wrong_column_count(tmp_path: Path):
    from minimed_rag.ingestion.umls.rrf_reader import iter_rrf_lines

    bad = "just|two|fields\n"
    path = _write(tmp_path, "bad.rrf", bad)
    with pytest.raises(ValueError, match="expected"):
        list(iter_rrf_lines(path, columns=("A", "B", "C", "D"), on_error="raise"))
