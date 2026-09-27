"""Local post-download filtering: NCDRC-only + award-language keyword check.

Runs entirely against already-fetched rows in raw_docs — it never triggers
an API call. That's the point: doctypes:consumer returns NCDRC plus State
and District commission matter (there is no ncdrc doctype in the API), so
NCDRC-only filtering has to happen here, on docsource, after download. And
because filtering is separate from fetching, filter logic can be tuned and
re-run indefinitely at zero additional cost — only the initial fetch spends
money.

matched_keywords is recorded for every doc (pass or fail) specifically so a
low pass rate is debuggable: you can inspect which award verbs/amounts did
or didn't fire, without re-fetching anything.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone

from ccpf.config import FilterConfig
from ccpf.models import FilterVerdict


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


class DocFilter:
    def __init__(self, category: str, config: FilterConfig):
        self.category = category
        self._ncdrc_pattern = re.compile(config.ncdrc_source_pattern, re.IGNORECASE)
        self._amount_patterns = _compile(config.award_amount_patterns)
        self._verb_patterns = _compile(config.award_verb_patterns)

    def evaluate(self, docsource: str, plain_text: str, tid: int) -> FilterVerdict:
        is_ncdrc = bool(self._ncdrc_pattern.search(docsource or ""))

        matched: list[str] = []
        amount_hit = False
        for pat in self._amount_patterns:
            m = pat.search(plain_text)
            if m:
                amount_hit = True
                matched.append(f"amount:{m.group(0)[:40]}")
                break

        verb_hit = False
        for pat in self._verb_patterns:
            m = pat.search(plain_text)
            if m:
                verb_hit = True
                matched.append(f"verb:{m.group(0)[:40]}")
                break

        has_award_language = amount_hit and verb_hit

        return FilterVerdict(
            tid=tid,
            category=self.category,
            is_ncdrc=is_ncdrc,
            has_award_language=has_award_language,
            matched_keywords=matched,
        )


def run_filters(conn: sqlite3.Connection, category: str, config: FilterConfig) -> list[FilterVerdict]:
    """Evaluate every fetched doc that hasn't been filtered for this category yet."""
    doc_filter = DocFilter(category, config)
    rows = conn.execute(
        """
        SELECT r.tid, r.docsource, r.plain_text
        FROM raw_docs r
        LEFT JOIN doc_filters f ON f.tid = r.tid AND f.category = ?
        WHERE f.tid IS NULL
        """,
        (category,),
    ).fetchall()

    verdicts = []
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        verdict = doc_filter.evaluate(row["docsource"], row["plain_text"], row["tid"])
        conn.execute(
            """
            INSERT INTO doc_filters
                (tid, category, is_ncdrc, has_award_language, matched_keywords, passed, filtered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tid, category) DO UPDATE SET
                is_ncdrc=excluded.is_ncdrc,
                has_award_language=excluded.has_award_language,
                matched_keywords=excluded.matched_keywords,
                passed=excluded.passed,
                filtered_at=excluded.filtered_at
            """,
            (
                verdict.tid,
                verdict.category,
                int(verdict.is_ncdrc),
                int(verdict.has_award_language),
                "|".join(verdict.matched_keywords),
                int(verdict.passed),
                now,
            ),
        )
        verdicts.append(verdict)
    conn.commit()
    return verdicts
