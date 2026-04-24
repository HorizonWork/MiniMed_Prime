"""Pseudocode PubMed ingestion pipeline."""
from __future__ import annotations

from biomed_kg.ingestion.base import SourceIngestionPipeline
from biomed_kg.ingestion.pubmed.normalizer import extract_pubmed_sections, normalize_pubmed_document
from biomed_kg.ingestion.pubmed.xml_parser import parse_pubmed_update_xml, parse_pubmed_xml


class PubMedIngestionPipeline(SourceIngestionPipeline):
    source_name = "pubmed"

    def __init__(self, downloader=None, pubmed_chunker=None, document_repo=None, **kwargs):
        super().__init__(**kwargs)
        self.downloader = downloader
        self.pubmed_chunker = pubmed_chunker
        self.document_repo = document_repo

    def download(self):
        return self.downloader.download_baseline_files("local") if self.downloader else []

    def run_baseline(self, release: str = "local") -> None:
        for xml_file in (self.downloader.download_baseline_files(release) if self.downloader else self.download()):
            self.write_raw(xml_file)
            for citation in parse_pubmed_xml(xml_file):
                doc = normalize_pubmed_document(citation)
                sections = extract_pubmed_sections(citation)
                chunks = [] if self.pubmed_chunker is None else [chunk for section in sections for chunk in self.pubmed_chunker.chunk(doc, [section])]
                if self.lakehouse:
                    self.lakehouse.write("document", [doc])
                    self.lakehouse.write("document_section", sections)
                    self.lakehouse.write("document_chunk", chunks)

    def run_daily_update(self, update_file) -> None:
        self.write_raw(update_file)
        for record in parse_pubmed_update_xml(update_file):
            if record.payload.get("status") == "deleted":
                if self.document_repo:
                    self.document_repo.mark_document_deleted(record.payload["pmid"])
                continue
            doc = normalize_pubmed_document(record)
            sections = extract_pubmed_sections(record)
            chunks = [] if self.pubmed_chunker is None else [chunk for section in sections for chunk in self.pubmed_chunker.chunk(doc, [section])]
            if self.document_repo:
                self.document_repo.upsert(doc.doc_id, doc)
                self.document_repo.replace_sections(doc.doc_id, sections)
                self.document_repo.replace_chunks(doc.doc_id, chunks)
