"""Hybrid retrieval: BM25 (keyword) + FAISS (dense vector), combined via
reciprocal rank fusion (RRF), with metadata pre-filtering.

RRF over each retriever's own rank — rather than trying to normalize and
weight BM25 scores against cosine similarities directly — sidesteps the
fact that the two score distributions aren't comparable: a BM25 score of
12 and a cosine similarity of 0.7 have no principled exchange rate, but
"ranked 3rd by keyword match, ranked 1st by semantic similarity" combines
cleanly. This is why hybrid search matters for this corpus specifically:
a query like "possession delay" needs BM25's exact phrase matching, while
"builder didn't hand over the flat on time" needs the embedding's semantic
similarity — neither retriever alone covers both phrasings well.

Metadata filtering (category/date/outcome) is applied as a genuine
PRE-filter — the candidate set is restricted before any scoring happens,
not scored-then-discarded after. At this corpus size (a few thousand
chunks) that means bypassing FAISS's ANN search and doing exhaustive
dot-product scoring over just the filtered subset via `reconstruct()`,
which is simpler and just as fast as approximate search would be here.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi


def reciprocal_rank_fusion(ranked_id_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    combined: dict[int, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            combined[chunk_id] = combined.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return combined


class HybridIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._chunk_ids: list[int] = []  # aligned with FAISS insertion position, _texts, _tokenized_corpus
        self._texts: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._pos_by_chunk_id: dict[int, int] = {}
        self._bm25: BM25Okapi | None = None

    def add(self, chunk_ids: list[int], embeddings: np.ndarray, texts: list[str]) -> None:
        assert len(chunk_ids) == embeddings.shape[0] == len(texts)
        start_pos = len(self._chunk_ids)
        self._faiss_index.add(embeddings)
        for offset, chunk_id in enumerate(chunk_ids):
            self._pos_by_chunk_id[chunk_id] = start_pos + offset
        self._chunk_ids.extend(chunk_ids)
        self._texts.extend(texts)
        self._tokenized_corpus.extend(t.lower().split() for t in texts)
        # Rebuilt on every add() rather than incrementally updated — cheap
        # at this corpus size, and avoids relying on rank_bm25's lack of an
        # incremental-update API.
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def search(
        self, query_embedding: np.ndarray, query_text: str, top_k: int = 50, min_dense_similarity: float | None = None
    ) -> list[tuple[int, float]]:
        return self.filtered_search(
            query_embedding, query_text, allowed_chunk_ids=None, top_k=top_k, min_dense_similarity=min_dense_similarity
        )

    def filtered_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        allowed_chunk_ids: set[int] | None,
        top_k: int = 50,
        min_dense_similarity: float | None = None,
    ) -> list[tuple[int, float]]:
        """`allowed_chunk_ids=None` searches the whole index; otherwise only
        chunks in that set are ever scored — a real pre-filter, not a
        post-hoc discard.

        `min_dense_similarity`, if given, drops candidates below that raw
        cosine similarity to the query from BOTH the dense and sparse
        ranking BEFORE fusion — not just the fused score. This is what
        makes a genuinely off-topic or meaningless query return nothing:
        without it, BM25/RRF can still assign a positive, plausible-looking
        score to the corpus's best-available (but unrelated) matches, since
        rank-based fusion has no notion of "none of these are actually
        relevant." Raw cosine similarity is bounded (-1 to 1) and doesn't
        depend on whether a reranker is used downstream, unlike a
        cross-encoder score or an RRF-fused score.
        """
        if allowed_chunk_ids is None:
            positions = list(range(len(self._chunk_ids)))
        else:
            positions = [self._pos_by_chunk_id[cid] for cid in allowed_chunk_ids if cid in self._pos_by_chunk_id]
        if not positions or self._bm25 is None:
            return []

        dense_scores = {
            self._chunk_ids[pos]: float(np.dot(self._faiss_index.reconstruct(pos), query_embedding))
            for pos in positions
        }

        if min_dense_similarity is not None:
            positions = [pos for pos in positions if dense_scores[self._chunk_ids[pos]] >= min_dense_similarity]
            if not positions:
                return []

        dense_ranked = [
            cid for cid, score in sorted(dense_scores.items(), key=lambda kv: kv[1], reverse=True)
            if min_dense_similarity is None or score >= min_dense_similarity
        ]

        bm25_scores_all = self._bm25.get_scores(query_text.lower().split())
        sparse_scores = {self._chunk_ids[pos]: float(bm25_scores_all[pos]) for pos in positions}
        sparse_ranked = [cid for cid, _ in sorted(sparse_scores.items(), key=lambda kv: kv[1], reverse=True)]

        fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked])
        return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    def text_for(self, chunk_id: int) -> str:
        return self._texts[self._pos_by_chunk_id[chunk_id]]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._faiss_index, str(path / "faiss.index"))
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(
                {
                    "dim": self.dim,
                    "chunk_ids": self._chunk_ids,
                    "texts": self._texts,
                    "tokenized_corpus": self._tokenized_corpus,
                    "pos_by_chunk_id": self._pos_by_chunk_id,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> "HybridIndex":
        with open(path / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        index = cls(dim=meta["dim"])
        index._faiss_index = faiss.read_index(str(path / "faiss.index"))
        index._chunk_ids = meta["chunk_ids"]
        index._texts = meta["texts"]
        index._tokenized_corpus = meta["tokenized_corpus"]
        index._pos_by_chunk_id = meta["pos_by_chunk_id"]
        index._bm25 = BM25Okapi(index._tokenized_corpus) if index._tokenized_corpus else None
        return index
