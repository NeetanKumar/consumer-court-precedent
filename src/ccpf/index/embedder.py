"""Embedder interface: local sentence-transformers by default, a hosted
implementation behind the same interface for later A/B comparison — the
locked-in decision from the project plan (local-first, hosted-swappable).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class LocalEmbedder:
    """CPU-only, no API cost, no network after the one-time model download."""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name or self.DEFAULT_MODEL)
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        self.dim = get_dim() if get_dim is not None else self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        # normalize_embeddings=True -> vectors are unit-length, so FAISS
        # inner-product search is equivalent to cosine similarity.
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True
        ).astype("float32")


class HostedEmbedder:
    """Stub for a hosted embeddings API (e.g. Voyage), behind the same
    interface as LocalEmbedder — not implemented for v1 per the plan's
    local-first decision. Swap in when comparing local vs. hosted quality."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "HostedEmbedder is not implemented for v1 (local-first decision in the project plan). "
            "Implement embed() against your chosen hosted API to use this."
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError
