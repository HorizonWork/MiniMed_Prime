from __future__ import annotations

from .base import Chunk, CorpusLoader


class StatPearlsLoader(CorpusLoader):
    """StatPearls is not available on HuggingFace (MedRAG/statpearls is empty).

    Use --corpus textbooks instead — MedRAG/textbooks contains 18 medical
    textbooks (Harrison's, Robbins, First Aid Step 1/2, ...) with the same
    clinical breadth (~125k chunks).
    """

    name = "statpearls"

    def load(self, limit: int = 0) -> list[Chunk]:
        raise RuntimeError(
            "MedRAG/statpearls is empty on HuggingFace.\n"
            "Use --corpus textbooks instead:\n"
            "  uv run minimed build-indexes bm25 --corpus textbooks --limit 500"
        )
