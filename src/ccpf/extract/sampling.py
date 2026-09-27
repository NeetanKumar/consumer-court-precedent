"""Stratified sampling of a validation set from already-extracted judgments.

Plain random sampling over a corpus dominated by one outcome (188/225 are
partly_allowed here) would mostly validate the easy, common case and give
almost no signal on the rare ones. This stratifies by (confidence, outcome)
instead:

  1. Every low-confidence extraction is included — these are exactly the
     ones most likely to be wrong.
  2. Medium-confidence extractions are included up to a cap.
  3. Remaining slots are filled from high-confidence extractions, split as
     evenly as possible across outcome categories (not proportionally) so
     the rare outcomes (allowed/dismissed) get real coverage instead of
     being drowned out by partly_allowed.

Sampling within each bucket is seeded for reproducibility.
"""
from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass


@dataclass
class SampleRow:
    tid: int
    plain_text: str
    confidence: str
    outcome: str


def stratified_sample(
    conn: sqlite3.Connection,
    category: str,
    sample_size: int = 40,
    medium_cap: int = 15,
    seed: int = 42,
) -> list[SampleRow]:
    rows = conn.execute(
        """
        SELECT e.tid, r.plain_text, e.confidence, e.judgment_json
        FROM extractions e
        JOIN raw_docs r ON r.tid = e.tid
        WHERE e.category = ?
        """,
        (category,),
    ).fetchall()

    pool = [
        SampleRow(
            tid=row["tid"],
            plain_text=row["plain_text"],
            confidence=row["confidence"],
            outcome=json.loads(row["judgment_json"])["outcome"],
        )
        for row in rows
    ]

    rng = random.Random(seed)

    low = [r for r in pool if r.confidence == "low"]
    medium = [r for r in pool if r.confidence == "medium"]
    high = [r for r in pool if r.confidence == "high"]

    selected: list[SampleRow] = list(low)

    rng.shuffle(medium)
    selected.extend(medium[:medium_cap])

    remaining = max(0, sample_size - len(selected))
    outcomes = sorted({r.outcome for r in high})
    by_outcome = {o: [r for r in high if r.outcome == o] for o in outcomes}
    for bucket in by_outcome.values():
        rng.shuffle(bucket)

    # Round-robin across outcome buckets so coverage is as even as
    # availability allows, rather than exhausting one bucket first.
    idx = {o: 0 for o in outcomes}
    while remaining > 0 and any(idx[o] < len(by_outcome[o]) for o in outcomes):
        for o in outcomes:
            if remaining <= 0:
                break
            if idx[o] < len(by_outcome[o]):
                selected.append(by_outcome[o][idx[o]])
                idx[o] += 1
                remaining -= 1

    return selected
