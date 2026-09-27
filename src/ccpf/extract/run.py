"""Resumable Stage 2 orchestrator: extract structured Judgments from
raw_docs that passed Stage 1 filters, for one category.

Same idempotency contract as ingest/run.py — every doc already in
`extractions` is skipped, so a crash or budget-cap stop costs nothing to
resume from.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import anthropic

from ccpf.extract.cost import LLMBudgetExceeded, LLMBudgetTracker
from ccpf.extract.extractor import ClaudeExtractor, ExtractionFailed

logger = logging.getLogger(__name__)


@dataclass
class ExtractionSummary:
    category: str
    attempted: int = 0
    succeeded: int = 0
    escalated: int = 0
    failed: int = 0
    skipped_existing: int = 0
    spend_start_microusd: int = 0
    spend_end_microusd: int = 0
    stopped_early: Optional[str] = None
    failures: list = field(default_factory=list)

    @property
    def spend_this_run_usd(self) -> float:
        return (self.spend_end_microusd - self.spend_start_microusd) / 1_000_000


def run_extraction(
    extractor: ClaudeExtractor,
    conn: sqlite3.Connection,
    budget: LLMBudgetTracker,
    category: str,
    max_docs: Optional[int] = None,
    tids: Optional[list[int]] = None,
    force: bool = False,
) -> ExtractionSummary:
    """
    `tids`, if given, restricts the run to that explicit set of documents —
    used to cheaply test a prompt change against a small, known subset (e.g.
    the validation set) before committing to a full re-extraction.

    `force`, combined with `tids`, re-extracts documents that already have
    an extraction row. The old row is replaced by an atomic UPSERT only
    once the new extraction succeeds — NOT by deleting upfront — so a
    mid-run failure (budget cap, API error) leaves the previous extraction
    intact instead of deleting it and coming up empty.
    """
    summary = ExtractionSummary(category=category, spend_start_microusd=budget.spent_microusd)

    if tids and force:
        # Select the named docs regardless of whether they're already
        # extracted — we intend to overwrite them.
        query = """
            SELECT r.tid, r.plain_text
            FROM raw_docs r
            JOIN doc_filters f ON f.tid = r.tid AND f.category = ?
            WHERE f.passed = 1
        """
        params: list = [category]
        query += f" AND r.tid IN ({','.join('?' * len(tids))})"
        params.extend(tids)
    else:
        query = """
            SELECT r.tid, r.plain_text
            FROM raw_docs r
            JOIN doc_filters f ON f.tid = r.tid AND f.category = ?
            LEFT JOIN extractions e ON e.tid = r.tid
            WHERE f.passed = 1 AND e.tid IS NULL
        """
        params = [category]
        if tids:
            query += f" AND r.tid IN ({','.join('?' * len(tids))})"
            params.extend(tids)

    rows = conn.execute(query, params).fetchall()

    for row in rows:
        if max_docs is not None and summary.attempted >= max_docs:
            break

        summary.attempted += 1
        try:
            result = extractor.extract(row["tid"], row["plain_text"])
        except LLMBudgetExceeded as exc:
            summary.stopped_early = str(exc)
            break
        except ExtractionFailed as exc:
            summary.failed += 1
            summary.failures.append(str(exc))
            continue
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
            # Account-level limits, rate limits, or transient API failures —
            # not something a retry loop should paper over here. Stop
            # cleanly so already-extracted rows (committed per-doc above)
            # are preserved and the run is resumable, same contract as
            # LLMBudgetExceeded and ingest/run.py's BudgetExceeded.
            summary.attempted -= 1  # this attempt never produced a result
            summary.stopped_early = f"Anthropic API error, stopping run: {exc}"
            logger.warning(summary.stopped_early)
            break

        conflict_clause = (
            """
            DO UPDATE SET
                category=excluded.category, model_used=excluded.model_used,
                escalated=excluded.escalated, confidence=excluded.confidence,
                judgment_json=excluded.judgment_json, extracted_at=excluded.extracted_at
            """
            if force
            else "DO NOTHING"
        )
        conn.execute(
            f"""
            INSERT INTO extractions (tid, category, model_used, escalated, confidence, judgment_json, extracted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tid) {conflict_clause}
            """,
            (
                row["tid"],
                category,
                result.model_used,
                int(result.escalated),
                result.judgment.confidence,
                result.judgment.model_dump_json(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        summary.succeeded += 1
        if result.escalated:
            summary.escalated += 1

    summary.spend_end_microusd = budget.spent_microusd
    return summary
