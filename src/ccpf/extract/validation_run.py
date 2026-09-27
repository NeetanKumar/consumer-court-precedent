"""Resumable orchestrator for building the independent validation set.

Same idempotency contract as extract/run.py: every tid already in
validation_labels is skipped, so a crash or budget-cap stop costs nothing
to resume from.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import anthropic

from ccpf.extract.cost import LLMBudgetExceeded, LLMBudgetTracker
from ccpf.extract.reviewer import IndependentReviewer, ReviewFailed
from ccpf.extract.sampling import SampleRow

logger = logging.getLogger(__name__)


@dataclass
class ValidationBuildSummary:
    category: str
    attempted: int = 0
    succeeded: int = 0
    skipped_existing: int = 0
    failed: int = 0
    spend_start_microusd: int = 0
    spend_end_microusd: int = 0
    stopped_early: Optional[str] = None
    failures: list = field(default_factory=list)

    @property
    def spend_this_run_usd(self) -> float:
        return (self.spend_end_microusd - self.spend_start_microusd) / 1_000_000


def build_validation_set(
    reviewer: IndependentReviewer,
    conn: sqlite3.Connection,
    budget: LLMBudgetTracker,
    category: str,
    sample: list[SampleRow],
) -> ValidationBuildSummary:
    summary = ValidationBuildSummary(category=category, spend_start_microusd=budget.spent_microusd)

    for row in sample:
        existing = conn.execute(
            "SELECT 1 FROM validation_labels WHERE tid = ? AND category = ?",
            (row.tid, category),
        ).fetchone()
        if existing is not None:
            summary.skipped_existing += 1
            continue

        summary.attempted += 1
        try:
            result = reviewer.review(row.tid, row.plain_text)
        except LLMBudgetExceeded as exc:
            summary.attempted -= 1
            summary.stopped_early = str(exc)
            break
        except ReviewFailed as exc:
            summary.failed += 1
            summary.failures.append(str(exc))
            continue
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            summary.attempted -= 1
            summary.stopped_early = f"Anthropic API error, stopping run: {exc}"
            logger.warning(summary.stopped_early)
            break

        conn.execute(
            """
            INSERT INTO validation_labels (tid, category, reviewer_model, gold_judgment_json, reviewed_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(tid, category) DO NOTHING
            """,
            (
                row.tid,
                category,
                reviewer.model,
                result.judgment.model_dump_json(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        summary.succeeded += 1

    summary.spend_end_microusd = budget.spent_microusd
    return summary
