from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from minimed_rag.retrieval.bm25_retriever import RetrievalResult

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class NoOpReranker:
    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return sorted(results, key=lambda item: item.score, reverse=True)


@dataclass(slots=True)
class CrossEncoderReranker:
    model_name: str = DEFAULT_RERANKER_MODEL
    batch_size: int = 16
    device: str | None = None
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)

    def rerank(self, query: str, results: list[RetrievalResult]) -> list[RetrievalResult]:
        if not results:
            return []

        self._ensure_model()
        import torch

        assert self._tokenizer is not None
        assert self._model is not None

        scores: list[float] = []
        pairs = [(query, result.chunk.text) for result in results]
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            queries = [item[0] for item in batch]
            passages = [item[1] for item in batch]
            encoded = self._tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self._model(**encoded).logits.detach().cpu().numpy()
            scores.extend(_logits_to_scores(logits).tolist())

        reranked = [
            RetrievalResult(chunk=result.chunk, score=float(score))
            for result, score in zip(results, scores, strict=True)
        ]
        return sorted(reranked, key=lambda item: item.score, reverse=True)

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(resolved_device)
        self._model.eval()


class Reranker(CrossEncoderReranker):
    """Backward-compatible default reranker name."""


def _logits_to_scores(logits: np.ndarray) -> np.ndarray:
    if logits.ndim == 1:
        return logits
    if logits.shape[1] == 1:
        return logits[:, 0]
    return logits[:, -1]


__all__ = ["CrossEncoderReranker", "DEFAULT_RERANKER_MODEL", "NoOpReranker", "Reranker"]
