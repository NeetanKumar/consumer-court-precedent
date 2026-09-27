"""Retrieval quality eval against the Stage 2 validation set.

We don't have true relevance judgments (a query mapped to *all* the
judgments that should count as relevant) — building that would need its
own manual/LLM labeling effort beyond the current scope. What we DO have
is the 40 independently gold-labeled judgments from Stage 2, each with a
`fact_summary` written independently by the reviewer model, in different
words than the extraction pipeline's own summary or the raw judgment text.

That gives a real, honest self-retrieval test: query the index with a
judgment's own (independently-written) fact summary, and check whether
the system retrieves that judgment back. This is NOT the same as recall@k
against multi-document relevance judgments — it only tells you "can the
system re-find a document from a paraphrase of it," not "does it find
every relevant precedent for a real user query." Both recall@k and MRR
terms are used here in that narrower, self-retrieval sense — labeled as
such in every report, not presented as general-purpose retrieval recall.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from ccpf.index.embedder import Embedder
from ccpf.index.retrieve import search
from ccpf.index.store import HybridIndex


@dataclass
class SelfRetrievalReport:
    n_queries: int
    recall_at_k: dict[int, float] = field(default_factory=dict)  # k -> fraction found within top-k
    mrr: float = 0.0
    misses: list[int] = field(default_factory=list)  # tids never found within the largest k tried


def evaluate_self_retrieval(
    conn: sqlite3.Connection,
    index: HybridIndex,
    embedder: Embedder,
    category: str,
    ks: tuple[int, ...] = (1, 5, 10),
) -> SelfRetrievalReport:
    rows = conn.execute(
        "SELECT tid, gold_judgment_json FROM validation_labels WHERE category = ?",
        (category,),
    ).fetchall()

    max_k = max(ks)
    reciprocal_ranks = []
    found_within = {k: 0 for k in ks}
    misses = []

    for row in rows:
        gold = json.loads(row["gold_judgment_json"])
        query_text = gold.get("fact_summary", "")
        if not query_text:
            continue

        results = search(conn, index, embedder, category, query_text, top_k=max_k)
        ranked_tids = [r.tid for r in results]

        rank = None
        if row["tid"] in ranked_tids:
            rank = ranked_tids.index(row["tid"]) + 1  # 1-based

        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        for k in ks:
            if rank is not None and rank <= k:
                found_within[k] += 1
        if rank is None:
            misses.append(row["tid"])

    n = len(reciprocal_ranks)
    return SelfRetrievalReport(
        n_queries=n,
        recall_at_k={k: (found_within[k] / n if n else 0.0) for k in ks},
        mrr=(sum(reciprocal_ranks) / n if n else 0.0),
        misses=misses,
    )
