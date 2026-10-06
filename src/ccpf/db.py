"""SQLite schema and connection helper.

Four tables. The first three (`search_pages`, `search_hits`, `raw_docs`) are
what make ingestion resumable at zero re-spend: every fetch loop in
ingest/run.py checks these tables *before* making a paid call and skips rows
that already exist. `spend_log` is an append-only ledger, read at startup to
recover cumulative spend across process restarts.

All money is stored as integer paise (1/100 rupee), never float rupees — a
float budget cap that drifts via repeated addition is a real way to overspend
against a metered paid API.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS search_pages (
    query_id    TEXT NOT NULL,
    pagenum     INTEGER NOT NULL,
    found       INTEGER NOT NULL,
    num_hits    INTEGER NOT NULL,
    fetched_at  TEXT NOT NULL,
    cost_paise  INTEGER NOT NULL,
    PRIMARY KEY (query_id, pagenum)
);

CREATE TABLE IF NOT EXISTS search_hits (
    tid         INTEGER PRIMARY KEY,
    query_id    TEXT NOT NULL,
    title       TEXT NOT NULL,
    headline    TEXT NOT NULL DEFAULT '',
    docsource   TEXT NOT NULL DEFAULT '',
    docsize     INTEGER,
    first_seen  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_docs (
    tid          INTEGER PRIMARY KEY,
    title        TEXT NOT NULL DEFAULT '',
    docsource    TEXT NOT NULL DEFAULT '',
    publishdate  TEXT NOT NULL DEFAULT '',
    raw_html     TEXT NOT NULL DEFAULT '',
    plain_text   TEXT NOT NULL DEFAULT '',
    fetched_at   TEXT NOT NULL,
    cost_paise   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS doc_filters (
    tid                INTEGER NOT NULL,
    category           TEXT NOT NULL,
    is_ncdrc           INTEGER NOT NULL,
    has_award_language INTEGER NOT NULL,
    matched_keywords   TEXT NOT NULL DEFAULT '[]',
    passed             INTEGER NOT NULL,
    filtered_at        TEXT NOT NULL,
    PRIMARY KEY (tid, category)
);

CREATE TABLE IF NOT EXISTS spend_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    endpoint    TEXT NOT NULL,
    ref         TEXT NOT NULL DEFAULT '',
    cost_paise  INTEGER NOT NULL,
    run_id      TEXT NOT NULL
);

-- Stage 2 (extraction). One row per successfully extracted judgment.
-- judgment_json is the validated Judgment model, serialized.
CREATE TABLE IF NOT EXISTS extractions (
    tid            INTEGER PRIMARY KEY,
    category       TEXT NOT NULL,
    model_used     TEXT NOT NULL,
    escalated      INTEGER NOT NULL,
    confidence     TEXT NOT NULL,
    judgment_json  TEXT NOT NULL,
    extracted_at   TEXT NOT NULL
);

-- Append-only LLM spend ledger, in integer MICRO-DOLLARS (1e-6 USD) — same
-- reasoning as spend_log's paise: a float USD cap that drifts via repeated
-- addition is a real way to overspend against a metered paid API.
CREATE TABLE IF NOT EXISTS llm_spend_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    model             TEXT NOT NULL,
    ref               TEXT NOT NULL DEFAULT '',
    input_tokens      INTEGER NOT NULL,
    output_tokens     INTEGER NOT NULL,
    cost_microusd     INTEGER NOT NULL,
    run_id            TEXT NOT NULL
);

-- Separate ledger from llm_spend_log — that one tracks the one-time,
-- offline Stage 2 pipeline cost (extraction + validation review). This
-- tracks live, per-query narration spend from the deployed chat app,
-- which needs its own budget cap: sharing llm_spend_log's cap would mean
-- narration inherits however much of that cap the offline pipeline
-- already used, which has nothing to do with how much narration should
-- be allowed to cost.
CREATE TABLE IF NOT EXISTS narration_spend_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT NOT NULL,
    model             TEXT NOT NULL,
    ref               TEXT NOT NULL DEFAULT '',
    input_tokens      INTEGER NOT NULL,
    output_tokens     INTEGER NOT NULL,
    cost_microusd     INTEGER NOT NULL
);

-- Independent gold-label review, used to measure Stage 2 extraction
-- accuracy. Produced by a DIFFERENT (stronger) model than the Haiku/Sonnet
-- pair that does extraction, blind to the extraction output, to avoid
-- correlated errors — see extract/reviewer.py.
CREATE TABLE IF NOT EXISTS validation_labels (
    tid                INTEGER NOT NULL,
    category           TEXT NOT NULL,
    reviewer_model     TEXT NOT NULL,
    gold_judgment_json TEXT NOT NULL,
    reviewed_at        TEXT NOT NULL,
    PRIMARY KEY (tid, category)
);

-- Stage 3 (indexing). Section-aware chunks of judgment text — the retrieval
-- unit. `chunk_index` is the chunk's position within its judgment;
-- `section` is a heuristic tag (facts | findings | operative_order).
-- Embeddings/FAISS/BM25 live on disk (data/index/<category>/), not here —
-- SQLite isn't a good fit for vector blobs at search time. `id` here is the
-- stable key that ties a disk-index row back to its text and metadata.
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tid          INTEGER NOT NULL,
    category     TEXT NOT NULL,
    section      TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    UNIQUE (tid, category, chunk_index)
);

-- Stage 5 (eval/observability). One row per answer-generation call: what
-- was asked, how long it took, and whether it refused. Cost is not
-- tracked here because Stage 3/4 queries make no paid API calls — that's
-- itself a notable property of the local-embeddings, non-LLM-narrated
-- design, not an oversight.
CREATE TABLE IF NOT EXISTS query_log (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT NOT NULL,
    category           TEXT NOT NULL,
    query_text         TEXT NOT NULL,
    filters_json       TEXT NOT NULL,
    sample_size        INTEGER NOT NULL,
    refused            INTEGER NOT NULL,
    latency_ms         REAL NOT NULL
);

-- Persistent cache: identical (category, query, filters) skips
-- recomputation. Keyed by a hash of the request, not the raw text, so the
-- key stays a fixed size regardless of query length.
CREATE TABLE IF NOT EXISTS query_cache (
    cache_key    TEXT PRIMARY KEY,
    answer_json  TEXT NOT NULL,
    cached_at    TEXT NOT NULL
);

-- Chat history for the Streamlit app. `owner` is a random id the browser
-- carries in the URL, so the sidebar only lists that visitor's own chats.
-- It is a capability token, not authentication.
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    owner       TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    answer_json      TEXT,
    feedback         INTEGER,  -- 1 = thumbs up, 0 = thumbs down, NULL = none
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner, updated_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_conv ON chat_messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_search_hits_query ON search_hits(query_id);
CREATE INDEX IF NOT EXISTS idx_spend_log_run ON spend_log(run_id);
CREATE INDEX IF NOT EXISTS idx_llm_spend_log_run ON llm_spend_log(run_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tid ON chunks(tid);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


@contextmanager
def connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def total_spend_paise(conn: sqlite3.Connection) -> int:
    """Cumulative spend across all runs — survives process restarts by design."""
    row = conn.execute("SELECT COALESCE(SUM(cost_paise), 0) AS total FROM spend_log").fetchone()
    return int(row["total"])


def has_search_page(conn: sqlite3.Connection, query_id: str, pagenum: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM search_pages WHERE query_id = ? AND pagenum = ?",
        (query_id, pagenum),
    ).fetchone()
    return row is not None


def has_raw_doc(conn: sqlite3.Connection, tid: int) -> bool:
    row = conn.execute("SELECT 1 FROM raw_docs WHERE tid = ?", (tid,)).fetchone()
    return row is not None


def total_llm_spend_microusd(conn: sqlite3.Connection) -> int:
    """Cumulative Stage 2 (offline pipeline) LLM spend, in micro-dollars — survives restarts."""
    row = conn.execute("SELECT COALESCE(SUM(cost_microusd), 0) AS total FROM llm_spend_log").fetchone()
    return int(row["total"])


def total_narration_spend_microusd(conn: sqlite3.Connection) -> int:
    """Cumulative live chat-narration spend, in micro-dollars — a separate
    ledger/cap from total_llm_spend_microusd (see narration_spend_log)."""
    row = conn.execute("SELECT COALESCE(SUM(cost_microusd), 0) AS total FROM narration_spend_log").fetchone()
    return int(row["total"])
