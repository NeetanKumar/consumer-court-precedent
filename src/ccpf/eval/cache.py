"""Persistent query cache — identical (category, query, filters) skips
recomputation. Persistent (SQLite-backed) rather than in-memory because
each CLI invocation is a fresh process; an in-memory cache would never
get a hit across separate `python scripts/answer.py` runs.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from ccpf.answer.schema import PrecedentAnswer

DEFAULT_TTL_HOURS = 24


def _cache_key(category: str, query_text: str, filters: dict, min_sample_size: int) -> str:
    # Sort keys so semantically-identical filter dicts with different
    # insertion order still hash to the same key.
    payload = json.dumps(
        {"category": category, "query": query_text, "filters": filters, "min_sample_size": min_sample_size},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached(
    conn: sqlite3.Connection,
    category: str,
    query_text: str,
    filters: dict,
    min_sample_size: int,
    ttl_hours: float = DEFAULT_TTL_HOURS,
) -> Optional[PrecedentAnswer]:
    key = _cache_key(category, query_text, filters, min_sample_size)
    row = conn.execute("SELECT answer_json, cached_at FROM query_cache WHERE cache_key = ?", (key,)).fetchone()
    if row is None:
        return None
    cached_at = datetime.fromisoformat(row["cached_at"])
    if datetime.now(timezone.utc) - cached_at > timedelta(hours=ttl_hours):
        return None
    return PrecedentAnswer.model_validate_json(row["answer_json"])


def store_cached(
    conn: sqlite3.Connection,
    category: str,
    query_text: str,
    filters: dict,
    min_sample_size: int,
    answer: PrecedentAnswer,
) -> None:
    key = _cache_key(category, query_text, filters, min_sample_size)
    conn.execute(
        """
        INSERT INTO query_cache (cache_key, answer_json, cached_at)
        VALUES (?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET answer_json=excluded.answer_json, cached_at=excluded.cached_at
        """,
        (key, answer.model_dump_json(), datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
