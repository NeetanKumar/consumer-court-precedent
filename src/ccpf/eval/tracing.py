"""Basic observability: log latency and outcome for every answer-generation
call. No cost field — Stage 3/4 queries make no paid API call, so cost per
query is always $0 here (unlike the LLM-narrated design the plan warned
against needing this kind of tracing for cost, not just latency).

Kept as a thin wrapper around generate_answer rather than instrumenting it
directly, so generate.py stays a pure function callers can test without
touching the DB's query_log table.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

from ccpf.answer.generate import MIN_RELEVANCE_SCORE, generate_answer
from ccpf.answer.schema import PrecedentAnswer
from ccpf.eval.cache import get_cached, store_cached
from ccpf.index.embedder import Embedder
from ccpf.index.rerank import CrossEncoderReranker
from ccpf.index.store import HybridIndex


def observed_generate_answer(
    conn: sqlite3.Connection,
    index: HybridIndex,
    embedder: Embedder,
    category: str,
    situation: str,
    outcomes: Optional[list[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_sample_size: int = 10,
    min_relevance_score: Optional[float] = MIN_RELEVANCE_SCORE,
    reranker: Optional[CrossEncoderReranker] = None,
    use_cache: bool = True,
) -> PrecedentAnswer:
    filters = {"outcomes": outcomes, "date_from": date_from, "date_to": date_to}
    # min_relevance_score only affects behavior when a reranker is passed
    # (see generate_answer) — fold that into the cache key so a call with a
    # reranker never serves a cached answer computed without one, or vice
    # versa, or at a different relevance floor.
    cache_extra = f"{min_relevance_score}|{reranker is not None}"

    t0 = time.perf_counter()
    if use_cache:
        cached = get_cached(conn, category, situation, {**filters, "_cache_extra": cache_extra}, min_sample_size)
        if cached is not None:
            _log(conn, category, situation, filters, cached, latency_ms=(time.perf_counter() - t0) * 1000)
            return cached

    answer = generate_answer(
        conn, index, embedder, category, situation,
        outcomes=outcomes, date_from=date_from, date_to=date_to,
        min_sample_size=min_sample_size, min_relevance_score=min_relevance_score, reranker=reranker,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    if use_cache:
        store_cached(conn, category, situation, {**filters, "_cache_extra": cache_extra}, min_sample_size, answer)

    _log(conn, category, situation, filters, answer, latency_ms)
    return answer


def _log(conn, category, situation, filters, answer: PrecedentAnswer, latency_ms: float) -> None:
    import json

    conn.execute(
        """
        INSERT INTO query_log (ts, category, query_text, filters_json, sample_size, refused, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            category,
            situation,
            json.dumps(filters),
            answer.sample_size,
            int(answer.refused),
            latency_ms,
        ),
    )
    conn.commit()
