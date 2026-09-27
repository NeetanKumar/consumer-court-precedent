"""Cross-encoder reranking of the hybrid retriever's candidates.

A cross-encoder scores (query, candidate) pairs jointly — much more
accurate than the bi-encoder similarity used for the initial retrieval,
but too slow to run over the whole corpus. The standard pattern: cheap
hybrid retrieval narrows thousands of chunks to ~50 candidates, then the
cross-encoder re-scores just those before picking the final top-N.
"""
from __future__ import annotations

import numpy as np


class CrossEncoderReranker:
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name or self.DEFAULT_MODEL)

    def rerank(self, query: str, candidates: list[tuple[int, str]], top_k: int = 10) -> list[tuple[int, float]]:
        """`candidates`: (chunk_id, text) pairs from hybrid retrieval. Returns
        (chunk_id, score) sorted best-first, truncated to top_k."""
        if not candidates:
            return []
        pairs = [(query, text) for _, text in candidates]
        scores = self._model.predict(pairs)
        order = np.argsort(-scores)[:top_k]
        return [(candidates[i][0], float(scores[i])) for i in order]
