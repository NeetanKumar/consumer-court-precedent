"""Query interface: metadata pre-filter -> hybrid retrieval -> cross-encoder
rerank -> aggregate chunk-level results back to judgment level.

Judgment-level aggregation matters for this project's actual use case
(compensation comparison across similar disputes, not "find the passage
that answers this question") — a user wants a handful of comparable
judgments, not five chunks that all happen to come from the same one.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Optional

from ccpf.index.embedder import Embedder
from ccpf.index.rerank import CrossEncoderReranker
from ccpf.index.store import HybridIndex

# Raw cosine similarity floor between the query embedding and the best-
# matching chunk. Verified empirically against this corpus/embedding model
# (all-MiniLM-L6-v2): gibberish and near-meaningless queries ("qsq", "23
# lac") top out around 0.26-0.34 cosine similarity even against their best
# available match, while a real situation description scores ~0.68-0.72.
# 0.4 sits with margin above the noise ceiling and well below real matches.
#
# This is the PRIMARY relevance floor, applied here (not only in
# answer/generate.py) so it protects every caller of retrieve.search() —
# including scripts/search.py directly — and works whether or not a
# reranker is used downstream. Rank-based fusion (RRF) and cross-encoder
# scores are NOT reliable substitutes for this: RRF always assigns a
# positive score to the corpus's best-available candidates regardless of
# whether any of them are actually relevant, and the cross-encoder floor
# in generate.py only applies when reranking is turned on.
MIN_DENSE_SIMILARITY = 0.4


@dataclass
class RetrievedJudgment:
    tid: int
    score: float
    best_chunk_text: str
    best_chunk_section: str
    judgment: dict  # the Stage 2 extracted Judgment, as a dict


def _allowed_tids(
    conn: sqlite3.Connection,
    category: str,
    outcomes: Optional[list[str]],
    date_from: Optional[str],
    date_to: Optional[str],
    duration_range: Optional[tuple[int, int]] = None,
) -> dict[int, dict]:
    """Metadata pre-filter: which judgments match, and their structured
    Judgment fields (needed later for the final result and for outcome
    filtering, since outcome lives in judgment_json, not a plain column)."""
    sql = """
        SELECT e.tid, e.judgment_json, r.publishdate
        FROM extractions e
        JOIN raw_docs r ON r.tid = e.tid
        WHERE e.category = ?
    """
    params: list = [category]
    if date_from:
        sql += " AND r.publishdate >= ?"
        params.append(date_from)
    if date_to:
        sql += " AND r.publishdate <= ?"
        params.append(date_to)

    rows = conn.execute(sql, params).fetchall()
    result = {}
    for row in rows:
        judgment = json.loads(row["judgment_json"])
        if outcomes is not None and judgment["outcome"] not in outcomes:
            continue
        if duration_range is not None:
            months = judgment.get("dispute_duration_months")
            # A facet means "comparable on this dimension", so records where
            # the duration is unknown cannot qualify.
            if months is None or not (duration_range[0] <= months <= duration_range[1]):
                continue
        result[row["tid"]] = judgment
    return result


def _chunk_id_to_tid(conn: sqlite3.Connection, category: str) -> dict[int, int]:
    rows = conn.execute("SELECT id, tid FROM chunks WHERE category = ?", (category,)).fetchall()
    return {row["id"]: row["tid"] for row in rows}


def search(
    conn: sqlite3.Connection,
    index: HybridIndex,
    embedder: Embedder,
    category: str,
    query_text: str,
    top_k: int = 10,
    outcomes: Optional[list[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    hybrid_candidates: int = 50,
    reranker: Optional[CrossEncoderReranker] = None,
    min_dense_similarity: Optional[float] = MIN_DENSE_SIMILARITY,
    duration_range: Optional[tuple[int, int]] = None,
) -> list[RetrievedJudgment]:
    judgments_by_tid = _allowed_tids(conn, category, outcomes, date_from, date_to, duration_range)
    if not judgments_by_tid:
        return []

    chunk_to_tid = _chunk_id_to_tid(conn, category)
    allowed_chunk_ids = {cid for cid, tid in chunk_to_tid.items() if tid in judgments_by_tid}
    if not allowed_chunk_ids:
        return []

    query_embedding = embedder.embed([query_text])[0]
    hybrid_results = index.filtered_search(
        query_embedding, query_text, allowed_chunk_ids=allowed_chunk_ids, top_k=hybrid_candidates,
        min_dense_similarity=min_dense_similarity,
    )
    if not hybrid_results:
        return []

    if reranker is not None:
        candidates = [(cid, index.text_for(cid)) for cid, _ in hybrid_results]
        reranked = reranker.rerank(query_text, candidates, top_k=len(candidates))
    else:
        reranked = hybrid_results

    # Aggregate to judgment level: keep each judgment's single best-scoring
    # chunk, in reranked order.
    best_by_tid: dict[int, tuple[int, float]] = {}
    for chunk_id, score in reranked:
        tid = chunk_to_tid[chunk_id]
        if tid not in best_by_tid or score > best_by_tid[tid][1]:
            best_by_tid[tid] = (chunk_id, score)

    ordered_tids = sorted(best_by_tid.items(), key=lambda kv: kv[1][1], reverse=True)[:top_k]

    results = []
    for tid, (chunk_id, score) in ordered_tids:
        section_row = conn.execute("SELECT section FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        results.append(
            RetrievedJudgment(
                tid=tid,
                score=score,
                best_chunk_text=index.text_for(chunk_id),
                best_chunk_section=section_row["section"] if section_row else "unknown",
                judgment=judgments_by_tid[tid],
            )
        )
    return results
