"""Streamlit chat UI for the Consumer Court Precedent Finder.

Run with: streamlit run app.py

Every number in an answer is computed deterministically from structured
judgment records (Stage 3/4, no LLM); Haiku only narrates that
already-computed result, streamed token by token. If narration fails or its
budget is exhausted, a deterministic headline is shown instead — the
metric cards, chart and citations below it are identical either way.

State lives in SQLite (ccpf.ui.store), not in st.session_state, so a page
refresh or a shared link keeps the conversation. Visitors are scoped by a
random `u` id carried in the URL (a capability token, not a login).

Follow-ups ("what interest rate?") are classified against the last
successful answer first and answered from that data without a new
retrieval; anything else runs the full pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import anthropic
import streamlit as st

from ccpf.answer.followup import answer_followup_stream, classify_message
from ccpf.answer.narrate import (
    NarrationBudgetExceeded,
    NarrationBudgetTracker,
    NarrationFailed,
    narrate_answer_stream,
)
from ccpf.answer.schema import PrecedentAnswer
from ccpf.config import DEFAULT_DB_PATH, REPO_ROOT, get_settings, load_app_config
from ccpf.db import connection
from ccpf.eval.tracing import observed_generate_answer
from ccpf.index.embedder import LocalEmbedder
from ccpf.index.rerank import CrossEncoderReranker
from ccpf.index.store import HybridIndex
from ccpf.ui import store
from ccpf.ui.limits import DISCLAIMER, MAX_MESSAGE_CHARS, SessionRateLimiter
from ccpf.ui.render import render_answer_body, render_refusal
from ccpf.ui.suggestions import follow_up_suggestions

DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "index"
CATEGORY = "builder_delay"  # only category built so far

EXAMPLES = [
    (":material/home_work:", "Builder delayed possession by 3 years and refused to refund the booking amount"),
    (":material/event_busy:", "Booked a flat in 2015, still not handed over 6 years later, seeking compensation"),
    (":material/payments:", "Developer cancelled my booking and won't return the advance payment"),
]

st.set_page_config(page_title="Consumer Court Precedent Finder", page_icon=":material/balance:", layout="centered")


# --- resources -----------------------------------------------------------

@st.cache_resource
def load_resources(category: str):
    embedder = LocalEmbedder()
    index = HybridIndex.load(DEFAULT_INDEX_DIR / category)
    reranker = CrossEncoderReranker()
    return embedder, index, reranker


@st.cache_resource
def load_anthropic_client():
    key = get_settings().anthropic_api_key
    return anthropic.Anthropic(api_key=key) if key else None  # narration is optional


@st.cache_data(ttl=600)
def corpus_size(category: str) -> int:
    with connection(DEFAULT_DB_PATH) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM extractions WHERE category = ?", (category,)).fetchone()[0])


# --- session identity ----------------------------------------------------

def _init_session() -> None:
    if "owner" not in st.session_state:
        st.session_state.owner = st.query_params.get("u") or store.new_id()
    st.query_params["u"] = st.session_state.owner
    if "conv_id" not in st.session_state:
        st.session_state.conv_id = st.query_params.get("c")
    if "rate_limiter" not in st.session_state:
        st.session_state.rate_limiter = SessionRateLimiter(max_requests=10, window_s=600)
    st.session_state.setdefault("pending", None)


def _set_conversation(conv_id: Optional[str]) -> None:
    st.session_state.conv_id = conv_id
    if conv_id:
        st.query_params["c"] = conv_id
    elif "c" in st.query_params:
        del st.query_params["c"]


# --- callbacks -----------------------------------------------------------

def _ask(text: str) -> None:
    st.session_state.pending = {"text": text, "add_user": True}


def _regenerate(user_text: str, assistant_id: int) -> None:
    with connection(DEFAULT_DB_PATH) as conn:
        store.delete_message(conn, st.session_state.owner, st.session_state.conv_id, assistant_id)
    st.session_state.pending = {"text": user_text, "add_user": False}


def _new_chat() -> None:
    _set_conversation(None)


def _open_chat(conv_id: str) -> None:
    _set_conversation(conv_id)


def _delete_chat(conv_id: str) -> None:
    with connection(DEFAULT_DB_PATH) as conn:
        store.delete_conversation(conn, st.session_state.owner, conv_id)
    if st.session_state.conv_id == conv_id:
        _set_conversation(None)


def _save_feedback(message_id: int) -> None:
    value = st.session_state.get(f"fb_{message_id}")
    with connection(DEFAULT_DB_PATH) as conn:
        store.set_feedback(conn, st.session_state.owner, message_id, value)


# --- rendering -----------------------------------------------------------

def _load_history() -> list[dict]:
    if not st.session_state.conv_id:
        return []
    with connection(DEFAULT_DB_PATH) as conn:
        return store.load_messages(conn, st.session_state.owner, st.session_state.conv_id)


def _answer_of(msg: dict) -> Optional[PrecedentAnswer]:
    return PrecedentAnswer.model_validate_json(msg["answer_json"]) if msg.get("answer_json") else None


def _last_successful_answer(history: list[dict]) -> Optional[PrecedentAnswer]:
    for msg in reversed(history):
        answer = _answer_of(msg) if msg["role"] == "assistant" else None
        if answer is not None and not answer.refused:
            return answer
    return None


def _render_assistant(msg: dict, user_text: Optional[str], is_last: bool) -> None:
    answer = _answer_of(msg)
    if answer is not None and answer.refused:
        render_refusal(answer)
        copy_text = answer.refusal_reason or ""
    else:
        st.markdown(msg["content"])
        if answer is not None:
            render_answer_body(answer)
        else:
            st.caption(DISCLAIMER)
        copy_text = msg["content"]

    with st.container(horizontal=True, vertical_alignment="center"):
        st.feedback(
            "thumbs", key=f"fb_{msg['id']}", default=msg.get("feedback"),
            on_change=_save_feedback, args=(msg["id"],),
        )
        with st.popover("", icon=":material/content_copy:", help="Copy this answer"):
            st.code(copy_text, language=None, wrap_lines=True)
        if is_last and user_text:
            st.button(
                "", icon=":material/refresh:", key=f"regen_{msg['id']}", help="Regenerate",
                type="tertiary", on_click=_regenerate, args=(user_text, msg["id"]),
            )

    if is_last and answer is not None:
        chips = follow_up_suggestions(answer)
        if chips:
            with st.container(horizontal=True):
                for i, chip in enumerate(chips):
                    st.button(chip, key=f"chip_{msg['id']}_{i}", on_click=_ask, args=(chip,), type="secondary")


def _render_history(history: list[dict]) -> None:
    last_user: Optional[str] = None
    last_assistant_idx = max((i for i, m in enumerate(history) if m["role"] == "assistant"), default=-1)
    for i, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                last_user = msg["content"]
                st.markdown(msg["content"])
            else:
                _render_assistant(msg, last_user, is_last=(i == last_assistant_idx))


def _render_empty_state() -> None:
    st.title("Consumer court precedent finder")
    st.markdown(
        "Describe a consumer dispute and see what similar complainants were actually awarded, "
        "with citations to the judgments."
    )
    for icon, text in EXAMPLES:
        st.button(f"{icon} {text}", key=f"ex_{text[:12]}", on_click=_ask, args=(text,), width="stretch")
    st.caption(f"Based on {corpus_size(CATEGORY)} NCDRC judgments on builder possession delay.")


def _render_sidebar() -> None:
    with st.sidebar:
        st.button("New chat", icon=":material/edit_square:", on_click=_new_chat, width="stretch")
        with connection(DEFAULT_DB_PATH) as conn:
            chats = store.list_conversations(conn, st.session_state.owner)
        if chats:
            st.caption("Recent")
        for chat in chats:
            with st.container(horizontal=True, vertical_alignment="center"):
                active = chat["id"] == st.session_state.conv_id
                st.button(
                    chat["title"], key=f"open_{chat['id']}", on_click=_open_chat, args=(chat["id"],),
                    type="primary" if active else "tertiary", width="stretch",
                )
                st.button(
                    "", icon=":material/delete:", key=f"del_{chat['id']}", help="Delete chat",
                    on_click=_delete_chat, args=(chat["id"],), type="tertiary",
                )
        st.divider()
        st.caption(
            "Every number is computed from structured judgment data, not generated. "
            "If fewer than 10 comparable judgments are found, the app declines to answer."
        )
        if load_anthropic_client() is None:
            st.caption(":orange[:material/info:] No API key set: answers show without the written summary.")


# --- answering -----------------------------------------------------------

def _try_followup(client, text: str, last_answer: PrecedentAnswer, status) -> Optional[str]:
    """Stream a follow-up answer from the last result's data. Returns the
    text, or None if this isn't a follow-up or answering failed (callers
    fall through to a fresh search)."""
    with connection(DEFAULT_DB_PATH) as conn:
        budget = NarrationBudgetTracker(conn, cap_usd=load_app_config().narration.budget_usd)
        status.update(label="Checking your previous result…")
        if classify_message(client, budget, text, last_answer) != "FOLLOWUP":
            return None
        slot = st.empty()
        try:
            with slot.container():
                return st.write_stream(answer_followup_stream(client, budget, text, last_answer))
        except (NarrationBudgetExceeded, NarrationFailed):
            slot.empty()
            return None


def _fallback_headline(answer: PrecedentAnswer) -> str:
    return f"Based on **{answer.sample_size} comparable judgments**, here is a compensation comparison for similar cases:"


def _answer_new_situation(client, text: str, status) -> tuple[str, Optional[PrecedentAnswer]]:
    status.update(label=f"Searching {corpus_size(CATEGORY)} judgments…")
    try:
        embedder, index, reranker = load_resources(CATEGORY)
    except FileNotFoundError:
        status.update(label="Search index not built", state="error")
        st.error("The search index hasn't been built yet. Run `python scripts/build_index.py --category builder_delay`.")
        return "", None

    with connection(DEFAULT_DB_PATH) as conn:
        answer = observed_generate_answer(conn, index, embedder, CATEGORY, text, reranker=reranker)
        if answer.refused:
            status.update(label="No close precedent found", state="complete")
            return "", answer

        status.update(label="Writing the answer…")
        headline = None
        if client is not None:
            budget = NarrationBudgetTracker(conn, cap_usd=load_app_config().narration.budget_usd)
            slot = st.empty()
            try:
                with slot.container():
                    headline = st.write_stream(narrate_answer_stream(client, budget, answer))
            except (NarrationBudgetExceeded, NarrationFailed):
                slot.empty()
        if not headline:
            headline = _fallback_headline(answer)
            st.markdown(headline)
    status.update(label=f"Found {answer.sample_size} comparable judgments", state="complete")
    return headline, answer


def _handle(text: str, add_user: bool) -> None:
    text = text.strip()[:MAX_MESSAGE_CHARS]
    if add_user and not st.session_state.rate_limiter.allow():
        wait = int(st.session_state.rate_limiter.retry_after_s()) + 1
        st.warning(f"You're sending messages too quickly. Please try again in about {wait}s.", icon=":material/hourglass_top:")
        return

    history = _load_history()
    last_answer = _last_successful_answer(history)

    with connection(DEFAULT_DB_PATH) as conn:
        if st.session_state.conv_id is None:
            _set_conversation(store.create_conversation(conn, st.session_state.owner, text))
        if add_user:
            store.add_message(conn, st.session_state.conv_id, "user", text)

    # Replay the earlier turns above the new exchange so the page doesn't jump.
    _render_history(history)
    if add_user:
        with st.chat_message("user"):
            st.markdown(text)

    client = load_anthropic_client()
    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=False)
        content: Optional[str] = None
        answer: Optional[PrecedentAnswer] = None

        if client is not None and last_answer is not None:
            content = _try_followup(client, text, last_answer, status)
            if content is not None:
                status.update(label="Answered from your previous result", state="complete")

        if content is None:
            content, answer = _answer_new_situation(client, text, status)
            if answer is None and not content:
                return  # error already shown

    with connection(DEFAULT_DB_PATH) as conn:
        store.add_message(
            conn, st.session_state.conv_id, "assistant", content or "",
            answer_json=answer.model_dump_json() if answer is not None else None,
        )
    st.rerun()


# --- main ----------------------------------------------------------------

_init_session()
_render_sidebar()

history = _load_history()
pending = st.session_state.pending
st.session_state.pending = None

typed = st.chat_input("Describe your situation…", max_chars=MAX_MESSAGE_CHARS)

if typed:
    _handle(typed, add_user=True)
elif pending:
    _handle(pending["text"], add_user=pending["add_user"])
elif history:
    _render_history(history)
else:
    _render_empty_state()
