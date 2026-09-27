"""Builds the Stage 3 index for one category: chunk generation (idempotent,
stored in SQLite) followed by embedding + hybrid index construction
(rebuilt fresh each run — cheap and fast at this corpus size, unlike the
paid Stage 1/2 calls, so there's no cost reason to make this incremental).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ccpf.index.chunking import chunk_judgment
from ccpf.index.embedder import Embedder
from ccpf.index.store import HybridIndex


@dataclass
class BuildSummary:
    category: str
    docs_chunked: int
    chunks_created: int
    chunks_total: int
    index_size: int


def generate_chunks(conn: sqlite3.Connection, category: str) -> tuple[int, int]:
    """Chunk every doc that passed Stage 1 filters and hasn't been chunked
    yet for this category. Idempotent — re-running only chunks new docs."""
    rows = conn.execute(
        """
        SELECT r.tid, r.plain_text
        FROM raw_docs r
        JOIN doc_filters f ON f.tid = r.tid AND f.category = ?
        LEFT JOIN chunks c ON c.tid = r.tid AND c.category = ?
        WHERE f.passed = 1 AND c.tid IS NULL
        """,
        (category, category),
    ).fetchall()

    docs_chunked = 0
    chunks_created = 0
    for row in rows:
        doc_chunks = chunk_judgment(row["tid"], row["plain_text"])
        for chunk in doc_chunks:
            conn.execute(
                """
                INSERT INTO chunks (tid, category, section, chunk_index, text)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tid, category, chunk_index) DO NOTHING
                """,
                (chunk.tid, category, chunk.section, chunk.chunk_index, chunk.text),
            )
            chunks_created += 1
        docs_chunked += 1
    conn.commit()
    return docs_chunked, chunks_created


def build_index(conn: sqlite3.Connection, category: str, embedder: Embedder, index_dir: Path) -> BuildSummary:
    docs_chunked, chunks_created = generate_chunks(conn, category)

    rows = conn.execute(
        "SELECT id, text FROM chunks WHERE category = ? ORDER BY id",
        (category,),
    ).fetchall()
    chunk_ids = [row["id"] for row in rows]
    texts = [row["text"] for row in rows]

    index = HybridIndex(dim=embedder.dim)
    if texts:
        # Embedding is batched to keep memory bounded (a habit worth
        # keeping if the corpus grows to other categories), but index.add()
        # is called once with everything — it rebuilds BM25 from scratch
        # on every call, so batching that too would just redo the same
        # work repeatedly for no benefit at this corpus size.
        batch_size = 64
        embeddings = np.concatenate(
            [embedder.embed(texts[start : start + batch_size]) for start in range(0, len(texts), batch_size)],
            axis=0,
        )
        index.add(chunk_ids, embeddings, texts)

    index.save(index_dir / category)

    return BuildSummary(
        category=category,
        docs_chunked=docs_chunked,
        chunks_created=chunks_created,
        chunks_total=len(chunk_ids),
        index_size=len(chunk_ids),
    )
