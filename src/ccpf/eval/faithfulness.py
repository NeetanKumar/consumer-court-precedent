"""Structural faithfulness verification for a Stage 4 PrecedentAnswer.

Classic RAGAS-style faithfulness checks whether an LLM's generated prose
is hallucinating claims not supported by its retrieved sources. Stage 4
deliberately has no LLM narrator — the answer is computed directly from
structured DB fields — so there's no free-text prose to hallucination-check.
The equivalent, honest check here is structural: does every number the
answer reports actually match the real data it claims to be derived from?
This can catch real bugs (a stats computation drifting from the DB, a
serialization round-trip corrupting a value) that a "looks plausible" read
of the answer would never catch.
"""
from __future__ import annotations

import json
import sqlite3
import statistics
from dataclasses import dataclass, field

from ccpf.answer.schema import PrecedentAnswer


@dataclass
class FaithfulnessReport:
    checked_stats: int = 0
    checked_citations: int = 0
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0


def _get_field(judgment: dict, field_name: str):
    if field_name == "amount_claimed":
        return judgment.get("amount_claimed")
    return (judgment.get("relief_components") or {}).get(field_name)


def verify_answer(conn: sqlite3.Connection, answer: PrecedentAnswer, category: str) -> FaithfulnessReport:
    report = FaithfulnessReport()
    if answer.refused:
        return report  # nothing to verify — a refusal makes no numeric claims

    stats_list = list(answer.relief_component_stats.values())
    if answer.amount_claimed_stats is not None:
        stats_list = [answer.amount_claimed_stats] + stats_list

    for stats in stats_list:
        report.checked_stats += 1
        values = []
        for tid in stats.contributing_tids:
            row = conn.execute(
                "SELECT judgment_json FROM extractions WHERE tid = ? AND category = ?", (tid, category)
            ).fetchone()
            if row is None:
                report.issues.append(f"{stats.field_name}: contributing tid={tid} has no extraction record")
                continue
            judgment = json.loads(row["judgment_json"])
            actual = _get_field(judgment, stats.field_name)
            if actual is None:
                report.issues.append(
                    f"{stats.field_name}: tid={tid} is listed as contributing but the DB value is null"
                )
                continue
            values.append(actual)

        if not values:
            continue

        if len(values) != stats.coverage:
            report.issues.append(
                f"{stats.field_name}: coverage claims {stats.coverage} but only {len(values)} "
                f"contributing tids have a non-null DB value"
            )

        expected_median = statistics.median(values)
        if stats.median is not None and abs(stats.median - expected_median) > 1e-6:
            report.issues.append(
                f"{stats.field_name}: reported median {stats.median} does not match "
                f"recomputed median {expected_median} from contributing_tids"
            )
        if stats.min is not None and abs(stats.min - min(values)) > 1e-6:
            report.issues.append(f"{stats.field_name}: reported min {stats.min} != recomputed {min(values)}")
        if stats.max is not None and abs(stats.max - max(values)) > 1e-6:
            report.issues.append(f"{stats.field_name}: reported max {stats.max} != recomputed {max(values)}")

    for citation in answer.citations:
        report.checked_citations += 1
        row = conn.execute(
            "SELECT judgment_json FROM extractions WHERE tid = ? AND category = ?", (citation.tid, category)
        ).fetchone()
        if row is None:
            report.issues.append(f"citation tid={citation.tid}: no extraction record found")
            continue
        judgment = json.loads(row["judgment_json"])
        if judgment.get("outcome") != citation.outcome:
            report.issues.append(
                f"citation tid={citation.tid}: outcome {citation.outcome!r} != DB outcome {judgment.get('outcome')!r}"
            )
        if judgment.get("amount_claimed") != citation.amount_claimed:
            report.issues.append(
                f"citation tid={citation.tid}: amount_claimed {citation.amount_claimed} != "
                f"DB value {judgment.get('amount_claimed')}"
            )

    return report
