"""Pure computation over a set of comparable judgments — no I/O, no LLM.

Deliberately separate from median/range over ALL retrieved judgments vs.
"coverage" for a specific field: sample_size is how many comparable
judgments were found; coverage is how many of those actually had a value
for e.g. refund. These differ a lot in practice — Stage 2's validation
found extraction sometimes leaves relief_components null even for
judgments the reviewer confirmed, so a median computed only over the
judgments that HAVE a value must say so explicitly, not silently imply it
covers the whole sample.
"""
from __future__ import annotations

import statistics
from typing import Optional

from ccpf.answer.schema import ComponentStats


def compute_component_stats(
    field_name: str,
    judgments: list[dict],
    getter,
    sample_size: int,
) -> ComponentStats:
    """`judgments`: list of {"tid": int, "judgment": dict}. `getter` extracts
    the numeric value (or None) from a judgment dict, e.g.
    `lambda j: j["relief_components"]["refund"]`."""
    values = []
    contributing_tids = []
    for row in judgments:
        val = getter(row["judgment"])
        if val is not None:
            values.append(val)
            contributing_tids.append(row["tid"])

    return ComponentStats(
        field_name=field_name,
        sample_size=sample_size,
        coverage=len(values),
        median=statistics.median(values) if values else None,
        min=min(values) if values else None,
        max=max(values) if values else None,
        contributing_tids=contributing_tids,
    )


def compute_award_counts(judgments: list[dict]) -> tuple[int, int]:
    """(awarded, known): how many judgments directed a monetary award, out
    of those where extraction could tell (award_made is not null). Old
    extractions predate the field, so known can be 0 — callers must treat
    that as "unknown", not "none awarded"."""
    known = [row["judgment"]["award_made"] for row in judgments if row["judgment"].get("award_made") is not None]
    return sum(1 for v in known if v), len(known)


def compute_outcome_distribution(judgments: list[dict]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for row in judgments:
        outcome = row["judgment"]["outcome"]
        dist[outcome] = dist.get(outcome, 0) + 1
    return dist
