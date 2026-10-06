"""Chat history persistence (conversations + messages + thumbs feedback).

Every read/write is scoped by `owner`, so one visitor can never list or
open another's conversation even though they share one SQLite file.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional

TITLE_MAX = 48


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_title(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= TITLE_MAX else text[: TITLE_MAX - 1].rstrip() + "…"


def create_conversation(conn: sqlite3.Connection, owner: str, first_message: str) -> str:
    cid = new_id()
    now = _now()
    conn.execute(
        "INSERT INTO conversations (id, owner, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (cid, owner, make_title(first_message), now, now),
    )
    conn.commit()
    return cid


def owns(conn: sqlite3.Connection, owner: str, conversation_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM conversations WHERE id = ? AND owner = ?", (conversation_id, owner)
    ).fetchone()
    return row is not None


def list_conversations(conn: sqlite3.Connection, owner: str, limit: int = 30) -> list[dict]:
    rows = conn.execute(
        "SELECT id, title, updated_at FROM conversations WHERE owner = ? ORDER BY updated_at DESC LIMIT ?",
        (owner, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def add_message(
    conn: sqlite3.Connection,
    conversation_id: str,
    role: str,
    content: str,
    answer_json: Optional[str] = None,
) -> int:
    now = _now()
    cur = conn.execute(
        "INSERT INTO chat_messages (conversation_id, role, content, answer_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (conversation_id, role, content, answer_json, now),
    )
    conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    conn.commit()
    return int(cur.lastrowid)


def load_messages(conn: sqlite3.Connection, owner: str, conversation_id: str) -> list[dict]:
    if not owns(conn, owner, conversation_id):
        return []
    rows = conn.execute(
        "SELECT id, role, content, answer_json, feedback FROM chat_messages WHERE conversation_id = ? ORDER BY id",
        (conversation_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def delete_message(conn: sqlite3.Connection, owner: str, conversation_id: str, message_id: int) -> None:
    if not owns(conn, owner, conversation_id):
        return
    conn.execute("DELETE FROM chat_messages WHERE id = ? AND conversation_id = ?", (message_id, conversation_id))
    conn.commit()


def set_feedback(conn: sqlite3.Connection, owner: str, message_id: int, value: Optional[int]) -> None:
    conn.execute(
        "UPDATE chat_messages SET feedback = ? WHERE id = ? AND role = 'assistant' AND conversation_id IN "
        "(SELECT id FROM conversations WHERE owner = ?)",
        (value, message_id, owner),
    )
    conn.commit()


def delete_conversation(conn: sqlite3.Connection, owner: str, conversation_id: str) -> None:
    if not owns(conn, owner, conversation_id):
        return
    conn.execute("DELETE FROM chat_messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
