from __future__ import annotations

from .base import Chunk, CorpusLoader

_DATA_FILES = "hf://datasets/MedRAG/textbooks/chunk/*.jsonl"

# 18 medical textbooks: Harrison's, Robbins, First Aid Step 1/2, Gray's Anatomy, etc.
# ~125k chunks total — good size for local BM25 index


class TextbooksLoader(CorpusLoader):
    """Load chunks from MedRAG/textbooks (medical textbooks curated for USMLE)."""

    name = "textbooks"

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
                    id=str(row.get("id", f"textbooks_{i}")),
                    text=text,
                    source="textbooks",
                    metadata={"title": title},
                )
            )
        return chunks
