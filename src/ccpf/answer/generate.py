"""Stage 4 orchestrator: retrieve comparable judgments (Stage 3), gate on
sample size, aggregate structured facts (Stage 2 output) into a precedent
comparison. No LLM call — the answer is computed directly from retrieved
structured records, so every number is traceable to specific judgments by
construction, not by an LLM's paraphrase of them.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

from ccpf.answer.aggregate import compute_component_stats, compute_outcome_distribution
from ccpf.answer.schema import Citation, PrecedentAnswer
from ccpf.index.embedder import Embedder
from ccpf.index.rerank import CrossEncoderReranker
from ccpf.index.retrieve import MIN_DENSE_SIMILARITY, search
from ccpf.index.store import HybridIndex

# Below this many comparable judgments, refuse rather than report a range —
# this is the "< ~10 comparable cases" refusal threshold from the plan.
MIN_SAMPLE_SIZE = 10

# Cross-encoder-score gate — DISABLED BY DEFAULT (None). This was tried as a
# secondary relevance floor on top of retrieve.MIN_DENSE_SIMILARITY, but a
# real query surfaced a false positive: "23 years since buying home... 3bhk
# house by broker... compensation amount" is genuinely on-topic — dense
# retrieval correctly found buyer's-agreement/possession-delay judgments,
# including one citing the landmark Pioneer Urban Land vs Govindan Raghavan
# possession-delay precedent — but the cross-encoder scored every one of
# them NEGATIVE, well below the 0.0 cutoff, triggering a false refusal.
# The cross-encoder's raw score scale was calibrated from exactly one
# example query and doesn't generalize across real phrasing diversity, so
# it's not reliable as a hard pass/fail gate — only retrieve.py's dense
# cosine-similarity floor (bounded, well-separated between noise ~0.34 and
# signal ~0.68, and correct on this exact case) is used for gating now.
# The cross-encoder still reorders results when a reranker is passed — it's
# just no longer trusted to decide whether to answer at all. Left
# available (not deleted) as an opt-in for anyone who wants to recalibrate
# it properly against a broader query set.
MIN_RELEVANCE_SCORE: Optional[float] = None

RELIEF_FIELDS = ["refund", "interest_rate", "mental_agony_compensation", "litigation_cost"]


def generate_answer(
    conn: sqlite3.Connection,
    index: HybridIndex,
    embedder: Embedder,
    category: str,
    situation: str,
    outcomes: Optional[list[str]] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sample_pool_size: int = 30,
    num_citations: int = 5,
    min_sample_size: int = MIN_SAMPLE_SIZE,
    min_relevance_score: Optional[float] = MIN_RELEVANCE_SCORE,
    min_dense_similarity: Optional[float] = MIN_DENSE_SIMILARITY,
    reranker: Optional[CrossEncoderReranker] = None,
) -> PrecedentAnswer:
    filters_applied = {"outcomes": outcomes, "date_from": date_from, "date_to": date_to}

    retrieved = search(
        conn, index, embedder, category, situation,
        top_k=sample_pool_size, outcomes=outcomes, date_from=date_from, date_to=date_to,
        reranker=reranker, min_dense_similarity=min_dense_similarity,
    )
    n_retrieved = len(retrieved)

    relevance_filtered = False
    if reranker is not None and min_relevance_score is not None:
        before = len(retrieved)
        retrieved = [r for r in retrieved if r.score >= min_relevance_score]
        relevance_filtered = len(retrieved) < before

    sample_size = len(retrieved)

    if sample_size < min_sample_size:
        if relevance_filtered:
            reason = (
                f"Only {sample_size} of {n_retrieved} retrieved judgment(s) were relevant enough "
                f"(cross-encoder score >= {min_relevance_score}) to this situation — below the minimum "
                f"of {min_sample_size} needed for a reliable compensation range. This usually means the "
                "situation described doesn't closely match this category's precedents, not that the "
                "category itself is thin."
            )
        else:
            reason = (
                f"Only {sample_size} comparable judgment(s) found for this situation and filters — "
                f"below the minimum of {min_sample_size} needed for a reliable compensation range. "
                "Insufficient precedent to answer confidently."
            )
        return PrecedentAnswer(
            query=situation,
            filters_applied=filters_applied,
            sample_size=sample_size,
            refused=True,
            refusal_reason=reason,
        )

    judgments = [{"tid": r.tid, "judgment": r.judgment} for r in retrieved]

    amount_claimed_stats = compute_component_stats(
        "amount_claimed", judgments, lambda j: j.get("amount_claimed"), sample_size
    )
    relief_stats = {
        field: compute_component_stats(
            field, judgments, lambda j, f=field: (j.get("relief_components") or {}).get(f), sample_size
        )
        for field in RELIEF_FIELDS
    }

    citations = [
        Citation(
            tid=r.tid,
            outcome=r.judgment["outcome"],
            amount_claimed=r.judgment.get("amount_claimed"),
            relief_components=r.judgment.get("relief_components") or {},
            fact_summary=r.judgment.get("fact_summary", ""),
            relevance_score=r.score,
        )
        for r in retrieved[:num_citations]
    ]

    return PrecedentAnswer(
        query=situation,
        filters_applied=filters_applied,
        sample_size=sample_size,
        refused=False,
        outcome_distribution=compute_outcome_distribution(judgments),
        amount_claimed_stats=amount_claimed_stats,
        relief_component_stats=relief_stats,
        citations=citations,
    )
