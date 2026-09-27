"""Resumable ingestion orchestrator for one category.

Every loop checks the DB for an existing row before making a paid call and
skips it if found — that's the entire resumability contract. Kill the
process at any point (Ctrl-C, crash, laptop sleep) and re-running picks up
exactly where it left off at zero re-spend, because search_pages and
raw_docs are the source of truth for "have we already paid for this."

Flow:
  1. Search phase: page through each configured query (0-based pagenum),
     recording hits into search_hits and marking pages done in
     search_pages. Stops when pagenum * page_size >= found.
  2. Doc phase: for every hit not yet in raw_docs, check the on-disk cache
     first (free) before calling the paid /doc/ endpoint.
  3. Filter phase: run local NCDRC + award-language filters over every
     fetched doc that hasn't been filtered for this category yet — no
     API calls, safe to re-run anytime.

Budget is enforced by BudgetTracker.check_and_reserve() before every
request in phases 1 and 2; BudgetExceeded stops the run cleanly with
whatever has been fetched so far intact and usable.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ccpf.config import CategoryConfig
from ccpf.ingest.budget import BudgetExceeded, BudgetTracker
from ccpf.ingest.cache import RawCache
from ccpf.ingest.client import IndianKanoonClient
from ccpf.ingest.filters import run_filters
from ccpf.ingest.htmlutil import html_to_text
from ccpf.ingest.queries import queries_for_category

logger = logging.getLogger(__name__)

SEARCH_PAGE_SIZE = 10  # Indian Kanoon search results per page (documented default).


@dataclass
class IngestSummary:
    category: str
    search_pages_fetched: int = 0
    docs_fetched: int = 0
    docs_from_cache: int = 0
    docs_skipped_existing: int = 0
    ncdrc_passed: int = 0
    award_passed: int = 0
    both_passed: int = 0
    spend_paise_start: int = 0
    spend_paise_end: int = 0
    stopped_early: Optional[str] = None
    hit_tids: set = field(default_factory=set)

    @property
    def spend_this_run_inr(self) -> float:
        return (self.spend_paise_end - self.spend_paise_start) / 100


def dry_run_projection(
    client: IndianKanoonClient,
    conn: sqlite3.Connection,
    category_name: str,
    category: CategoryConfig,
    budget: BudgetTracker,
) -> dict:
    """Costs ONE search call per query (page 0) to read `found`, projects the
    rest without fetching further pages or any documents."""
    projections = {}
    total_projected_paise = 0
    for q in queries_for_category(category):
        cost = budget.check_and_reserve("search", ref=f"{q.id}:0")
        resp = client.search(
            q.text, pagenum=0,
            on_success=lambda _data, c=cost, qid=q.id: budget.record_spend("search", c, ref=f"{qid}:0"),
        )
        _record_search_page(conn, q.id, 0, resp.found, len(resp.docs), cost)

        projected_search = budget.project_search_cost(resp.found, SEARCH_PAGE_SIZE)
        projected_docs = resp.found * budget.cost_of("doc")
        projections[q.id] = {
            "found": resp.found,
            "projected_search_paise": projected_search,
            "projected_doc_paise": projected_docs,
        }
        total_projected_paise += projected_search + projected_docs

    return {
        "queries": projections,
        "total_projected_inr": total_projected_paise / 100,
        "already_spent_inr": budget.spent_inr,
        "cap_inr": budget.cap_paise / 100,
    }


def _record_search_page(conn, query_id: str, pagenum: int, found: int, num_hits: int, cost_paise: int) -> None:
    conn.execute(
        """
        INSERT INTO search_pages (query_id, pagenum, found, num_hits, fetched_at, cost_paise)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(query_id, pagenum) DO NOTHING
        """,
        (query_id, pagenum, found, num_hits, datetime.now(timezone.utc).isoformat(), cost_paise),
    )
    conn.commit()


def run_ingestion(
    client: IndianKanoonClient,
    conn: sqlite3.Connection,
    cache: RawCache,
    category_name: str,
    category: CategoryConfig,
    budget: BudgetTracker,
    max_pages: Optional[int] = None,
    max_docs: Optional[int] = None,
) -> IngestSummary:
    summary = IngestSummary(category=category_name, spend_paise_start=budget.spent_paise)

    try:
        _search_phase(client, conn, category, budget, summary, max_pages)
        _doc_phase(client, conn, cache, budget, summary, max_docs)
    except BudgetExceeded as exc:
        summary.stopped_early = str(exc)
        logger.warning("Stopping ingestion: %s", exc)

    if category.filters is not None:
        verdicts = run_filters(conn, category_name, category.filters)
        for v in verdicts:
            summary.ncdrc_passed += int(v.is_ncdrc)
            summary.award_passed += int(v.has_award_language)
            summary.both_passed += int(v.passed)

    summary.spend_paise_end = budget.spent_paise
    return summary


def _search_phase(client, conn, category, budget, summary, max_pages) -> None:
    for q in queries_for_category(category):
        pagenum = 0
        pages_this_query = 0
        while True:
            if max_pages is not None and pages_this_query >= max_pages:
                break

            existing = conn.execute(
                "SELECT found, num_hits FROM search_pages WHERE query_id = ? AND pagenum = ?",
                (q.id, pagenum),
            ).fetchone()
            if existing is not None:
                # Already paid for this page — skip straight to the stop check.
                found, num_hits = existing["found"], existing["num_hits"]
                pages_this_query += 1
                if num_hits == 0 or (pagenum + 1) * SEARCH_PAGE_SIZE >= found:
                    break
                pagenum += 1
                continue

            cost = budget.check_and_reserve("search", ref=f"{q.id}:{pagenum}")
            ref = f"{q.id}:{pagenum}"
            resp = client.search(
                q.text, pagenum=pagenum,
                on_success=lambda _data, c=cost, r=ref: budget.record_spend("search", c, ref=r),
            )
            _record_search_page(conn, q.id, pagenum, resp.found, len(resp.docs), cost)
            summary.search_pages_fetched += 1

            now = datetime.now(timezone.utc).isoformat()
            for hit in resp.docs:
                summary.hit_tids.add(hit.tid)
                conn.execute(
                    """
                    INSERT INTO search_hits (tid, query_id, title, headline, docsource, docsize, first_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tid) DO NOTHING
                    """,
                    (hit.tid, q.id, hit.title, hit.headline, hit.docsource, hit.docsize, now),
                )
            conn.commit()

            pages_this_query += 1
            if len(resp.docs) == 0 or (pagenum + 1) * SEARCH_PAGE_SIZE >= resp.found:
                break
            pagenum += 1


def _doc_phase(client, conn, cache, budget, summary, max_docs) -> None:
    rows = conn.execute("SELECT DISTINCT tid FROM search_hits").fetchall()
    tids = [row["tid"] for row in rows]

    fetched_count = 0
    for tid in tids:
        if max_docs is not None and fetched_count >= max_docs:
            break

        existing = conn.execute("SELECT 1 FROM raw_docs WHERE tid = ?", (tid,)).fetchone()
        if existing is not None:
            summary.docs_skipped_existing += 1
            continue

        cached = cache.load(tid)
        if cached is not None:
            data = cached
            summary.docs_from_cache += 1
        else:
            cost = budget.check_and_reserve("doc", ref=str(tid))
            raw_holder: dict = {}
            client.doc(
                tid,
                on_success=lambda d, c=cost, t=tid: (raw_holder.update(d), budget.record_spend("doc", c, ref=str(t))),
            )
            data = raw_holder
            cache.save(tid, data)
            summary.docs_fetched += 1

        plain_text = html_to_text(data.get("doc", ""))
        conn.execute(
            """
            INSERT INTO raw_docs (tid, title, docsource, publishdate, raw_html, plain_text, fetched_at, cost_paise)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tid) DO NOTHING
            """,
            (
                tid,
                data.get("title", ""),
                data.get("docsource", ""),
                data.get("publishdate", ""),
                data.get("doc", ""),
                plain_text,
                datetime.now(timezone.utc).isoformat(),
                0 if cached is not None else budget.cost_of("doc"),
            ),
        )
        conn.commit()
        fetched_count += 1
