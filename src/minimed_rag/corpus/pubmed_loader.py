from __future__ import annotations

from .base import Chunk, CorpusLoader

_DATA_FILES = "hf://datasets/MedRAG/pubmed/chunk/*.jsonl"


class PubMedLoader(CorpusLoader):
    """Load abstract-level chunks from MedRAG/pubmed (1166 JSONL files in chunk/)."""

    name = "pubmed"

    def load(self, limit: int = 0) -> list[Chunk]:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("json", data_files=_DATA_FILES, split="train", streaming=True)
        chunks: list[Chunk] = []
        for i, row in enumerate(ds):
            if limit > 0 and i >= limit:
                break
            title = row.get("title") or ""
            content = row.get("content") or ""
            text = f"{title}\n{content}".strip() if title else content
            if not text:
                continue
            chunks.append(
                Chunk(
                    id=str(row.get("id", f"pubmed_{i}")),
                    text=text,
                    source="pubmed",
                    metadata={"title": title, "pmid": str(row.get("PMID", ""))},
                )
            )
        return chunks
