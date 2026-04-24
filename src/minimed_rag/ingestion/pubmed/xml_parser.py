"""Pseudocode PubMed XML parser."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from minimed_rag.ingestion.base import RawFile, RawRecord


def parse_pubmed_xml(raw_file: RawFile) -> list[RawRecord]:
    root = ET.fromstring(raw_file.bytes or b"<PubmedArticleSet />")
    return [
        RawRecord(
            "PubMedCitation",
            {"xml": ET.tostring(node, encoding="unicode"), "release": raw_file.release},
        )
        for node in root
    ]


def parse_pubmed_update_xml(raw_file: RawFile) -> list[RawRecord]:
    return parse_pubmed_xml(raw_file)
