"""Compares Stage 2 extraction output against the independent gold labels.

Field-level scoring, not a single blended number — a portfolio project's
credibility rests on being honest about *where* extraction is weak (e.g.
"interest rate accuracy is poor" is a more useful and more truthful
finding than "87% overall accuracy").
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

NUMERIC_FIELDS = [
    ("amount_claimed", None, 0.10),  # top-level, relative tolerance
    ("dispute_duration_months", None, None),  # exact-ish, handled separately (±2 months)
    ("relief_components.refund", "relief_components", 0.10),
    ("relief_components.interest_rate", "relief_components", 0.15),
    ("relief_components.mental_agony_compensation", "relief_components", 0.10),
    ("relief_components.litigation_cost", "relief_components", 0.10),
]


@dataclass
class FieldScore:
    field: str
    both_null: int = 0
    matched: int = 0
    mismatched: int = 0
    gold_only: int = 0  # gold had a value, extraction said null — a miss
    extraction_only: int = 0  # extraction guessed a value gold doesn't have

    @property
    def total_comparable(self) -> int:
        # Excludes both_null — agreeing there's no value isn't a meaningful signal.
        return self.matched + self.mismatched + self.gold_only + self.extraction_only

    @property
    def accuracy(self) -> float:
        if self.total_comparable == 0:
            return 1.0
        return self.matched / self.total_comparable


@dataclass
class ValidationReport:
    n: int
    outcome: FieldScore = field(default_factory=lambda: FieldScore("outcome"))
    numeric: dict = field(default_factory=dict)
    accuracy_by_confidence: dict = field(default_factory=dict)  # confidence -> field-agreement rate


def _get(judgment: dict, path: str):
    parts = path.split(".")
    val = judgment
    for p in parts:
        if val is None:
            return None
        val = val.get(p)
    return val


def _score_numeric(gold_val, ext_val, rel_tol: float | None, abs_tol: float | None, score: FieldScore) -> None:
    if gold_val is None and ext_val is None:
        score.both_null += 1
        return
    if gold_val is not None and ext_val is None:
        score.gold_only += 1
        return
    if gold_val is None and ext_val is not None:
        score.extraction_only += 1
        return
    if abs_tol is not None:
        ok = abs(gold_val - ext_val) <= abs_tol
    else:
        denom = max(abs(gold_val), 1e-9)
        ok = abs(gold_val - ext_val) / denom <= rel_tol
    if ok:
        score.matched += 1
    else:
        score.mismatched += 1


def score_validation_set(conn: sqlite3.Connection, category: str) -> ValidationReport:
    rows = conn.execute(
        """
        SELECT v.tid, v.gold_judgment_json, e.judgment_json, e.confidence
        FROM validation_labels v
        JOIN extractions e ON e.tid = v.tid AND e.category = v.category
        WHERE v.category = ?
        """,
        (category,),
    ).fetchall()

    report = ValidationReport(n=len(rows))
    report.numeric = {path: FieldScore(path) for path, _, _ in NUMERIC_FIELDS}

    conf_totals: dict[str, list[bool]] = {}

    for row in rows:
        gold = json.loads(row["gold_judgment_json"])
        ext = json.loads(row["judgment_json"])

        if gold["outcome"] == ext["outcome"]:
            report.outcome.matched += 1
            field_ok = True
        else:
            report.outcome.mismatched += 1
            field_ok = False

        for path, _, rel_tol in NUMERIC_FIELDS:
            score = report.numeric[path]
            gold_val = _get(gold, path)
            ext_val = _get(ext, path)
            if path == "dispute_duration_months":
                _score_numeric(gold_val, ext_val, rel_tol=None, abs_tol=2, score=score)
            else:
                _score_numeric(gold_val, ext_val, rel_tol=rel_tol, abs_tol=None, score=score)

        conf_totals.setdefault(row["confidence"], []).append(field_ok)

    report.accuracy_by_confidence = {
        conf: sum(vals) / len(vals) for conf, vals in conf_totals.items()
    }
    return report
