from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


PoolingMode = Literal["cls", "mean"]


@dataclass(slots=True)
class TransformerEmbedder:
    """Small local embedding wrapper for biomedical retrieval experiments.

    The model/tokenizer are loaded lazily so importing CLI modules stays cheap.
    """

    model_name: str = DEFAULT_EMBEDDING_MODEL
    batch_size: int = 32
    max_length: int = 512
    device: str | None = None
    normalize: bool = True
    pooling: PoolingMode | None = None
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.pooling is None:
            self.pooling = "cls" if "bge-" in self.model_name.lower() else "mean"

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into a float32 numpy matrix."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        self._ensure_model()

        import torch

        vectors: list[np.ndarray] = []
        assert self._tokenizer is not None
        assert self._model is not None

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._model.device) for key, value in encoded.items()}
            with torch.no_grad():
                output = self._model(**encoded)
                embedding = self._pool(output.last_hidden_state, encoded["attention_mask"])
                if self.normalize:
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            vectors.append(embedding.detach().cpu().numpy().astype(np.float32, copy=False))

        return np.vstack(vectors)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Encode queries, adding the BGE retrieval instruction when appropriate."""
        if "bge-" not in self.model_name.lower():
            return self.encode(texts)
        instructed = [f"Represent this sentence for searching relevant passages: {text}" for text in texts]
        return self.encode(instructed)

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModel, AutoTokenizer

        resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.to(resolved_device)
        self._model.eval()

    def _pool(self, token_embeddings, attention_mask):
        if self.pooling == "cls":
            return token_embeddings[:, 0]

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        summed = (token_embeddings * input_mask_expanded).sum(dim=1)
        counts = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return summed / counts


__all__ = ["DEFAULT_EMBEDDING_MODEL", "TransformerEmbedder"]
