"""Streamlit chat UI for the Consumer Court Precedent Finder.

Run with: streamlit run app.py

Two-stage answer: Stage 3/4 retrieval + aggregation computes every number
directly from Stage 2's structured judgment records (no LLM, fully
traceable) — then an LLM (Haiku 4.5) NARRATES that already-computed data
into conversational prose. The narration model never computes or invents
a figure; see answer/narrate.py for the faithfulness constraints on it.
If narration fails or its budget is exhausted, the app falls back to a
simple deterministic headline instead of losing the response entirely —
the metric cards, table, and citation cards below render identically
either way, so a fallback doesn't look degraded.

Deliberately no exposed filters/knobs (category, outcome, date range,
min sample size, rerank toggle, cache toggle, faithfulness check) — those
are internal tuning parameters, not something an end user should need to
understand.

Chat history stores each assistant turn as {headline, answer: dict | None}
rather than a flat markdown string — replay re-renders through the same
_render_answer() path as a live response, so metric cards and citation
cards look identical on reload, not just on the turn they were created.

Follow-up questions: without this, every message re-runs full retrieval
from scratch as if it were an independent new situation, so "what is the
median interest rate" right after a real answer gets searched for AS a
consumer dispute, finds nothing, and falsely refuses. A cheap classifier
(answer/followup.py) checks each new message against the last successful
answer first — a genuine follow-up is answered from that existing data
with no new retrieval; anything else runs the normal pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import anthropic
import streamlit as st

from ccpf.answer.followup import answer_followup, classify_message
from ccpf.answer.narrate import NarrationBudgetExceeded, NarrationBudgetTracker, NarrationFailed, format_inr, narrate_answer
from ccpf.answer.schema import Citation, ComponentStats, PrecedentAnswer
from ccpf.config import DEFAULT_DB_PATH, REPO_ROOT, get_settings, load_app_config
from ccpf.db import connection
from ccpf.eval.tracing import observed_generate_answer
from ccpf.index.embedder import LocalEmbedder
from ccpf.index.rerank import CrossEncoderReranker
from ccpf.index.store import HybridIndex

DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "index"
CATEGORY = "builder_delay"  # only category built so far

OUTCOME_BADGE = {
    "allowed": "🟢 Allowed",
    "partly_allowed": "🟠 Partly allowed",
    "dismissed": "🔴 Dismissed",
}

EXAMPLE_SITUATIONS = [
    "Builder delayed possession by 3 years and refused to refund the booking amount",
    "Booked a flat in 2015, still not handed over 6 years later, seeking compensation",
    "Developer cancelled my booking and won't return the advance payment",
]

st.set_page_config(page_title="Consumer Court Precedent Finder", page_icon="⚖️", layout="wide")


@st.cache_resource
def load_resources(category: str):
    embedder = LocalEmbedder()
    index = HybridIndex.load(DEFAULT_INDEX_DIR / category)
    reranker = CrossEncoderReranker()
    return embedder, index, reranker


@st.cache_resource
def load_anthropic_client():
    settings = get_settings()
    key = settings.anthropic_api_key
    if not key:
        return None  # narration is optional — app still works without it
    return anthropic.Anthropic(api_key=key)


def _format_value(stats: ComponentStats, value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if stats.field_name == "interest_rate":
        return f"{value:.1f}%"
    return format_inr(value)


def _format_citation_relief(c: Citation) -> str:
    """The relief actually awarded in this specific judgment — separate
    from the aggregated median/min/max stats, which blend across all
    cited judgments. Without this, a citation only showed what was
    claimed, never what was awarded."""
    parts = []
    if c.amount_claimed is not None:
        parts.append(f"claimed {format_inr(c.amount_claimed)}")
    refund = c.relief_components.get("refund")
    if refund is not None:
        parts.append(f"refund awarded {format_inr(refund)}")
    interest = c.relief_components.get("interest_rate")
    if interest is not None:
        parts.append(f"interest {interest:.1f}%")
    agony = c.relief_components.get("mental_agony_compensation")
    if agony is not None:
        parts.append(f"mental agony {format_inr(agony)}")
    litigation = c.relief_components.get("litigation_cost")
    if litigation is not None:
        parts.append(f"litigation cost {format_inr(litigation)}")
    return " · ".join(parts) if parts else "amounts not extracted for this judgment"


def _render_metrics(answer: PrecedentAnswer) -> None:
    cols = st.columns(3)
    cols[0].metric("Comparable judgments", answer.sample_size)

    refund_stats = answer.relief_component_stats.get("refund")
    if refund_stats and refund_stats.median is not None:
        cols[1].metric(
            "Median refund awarded", format_inr(refund_stats.median),
            help=f"Based on {refund_stats.coverage}/{refund_stats.sample_size} judgments with this data",
        )
    elif answer.amount_claimed_stats and answer.amount_claimed_stats.median is not None:
        cols[1].metric("Median amount claimed", format_inr(answer.amount_claimed_stats.median))

    interest_stats = answer.relief_component_stats.get("interest_rate")
    if interest_stats and interest_stats.median is not None:
        cols[2].metric(
            "Median interest rate", f"{interest_stats.median:.1f}%",
            help=f"Based on {interest_stats.coverage}/{interest_stats.sample_size} judgments with this data",
        )


def _render_citation_card(c: Citation) -> None:
    with st.container(border=True):
        badge = OUTCOME_BADGE.get(c.outcome, c.outcome)
        st.markdown(f"**tid {c.tid}**&nbsp;&nbsp;·&nbsp;&nbsp;{badge}&nbsp;&nbsp;·&nbsp;&nbsp;relevance {c.relevance_score:.2f}")
        st.caption(_format_citation_relief(c))
        summary = c.fact_summary[:250] + ("…" if len(c.fact_summary) > 250 else "")
        st.write(summary)


def _render_details(answer: PrecedentAnswer) -> None:
    with st.expander("📊 Full statistics & cited judgments"):
        st.markdown(
            "**Outcome distribution:** "
            + ", ".join(f"{OUTCOME_BADGE.get(k, k)}: {v}" for k, v in answer.outcome_distribution.items())
        )
        lines = ["| Field | Coverage | Median | Min | Max |", "|---|---|---|---|---|"]
        stats_list = [answer.amount_claimed_stats] if answer.amount_claimed_stats else []
        stats_list += list(answer.relief_component_stats.values())
        for s in stats_list:
            cov = f"{s.coverage}/{s.sample_size} ({s.coverage_fraction*100:.0f}%)"
            lines.append(f"| {s.field_name} | {cov} | {_format_value(s, s.median)} | {_format_value(s, s.min)} | {_format_value(s, s.max)} |")
        st.markdown("\n".join(lines))

        st.markdown("**Cited judgments:**")
        for c in answer.citations:
            _render_citation_card(c)


def _render_answer(headline: str, answer_dict: Optional[dict]) -> None:
    """Shared rendering path for both a freshly computed response and a
    replayed history entry — a refusal, a citation card, a metric all look
    the same whether they're brand new or re-rendered after a rerun."""
    if answer_dict is None:
        st.markdown(headline)
        return

    answer = PrecedentAnswer.model_validate(answer_dict)
    if answer.refused:
        st.warning(f"**Insufficient precedent.**\n\n{answer.refusal_reason}", icon="⚠️")
        return

    st.markdown(headline)
    _render_metrics(answer)
    _render_details(answer)


def _last_successful_answer() -> Optional[PrecedentAnswer]:
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and msg.get("answer") is not None:
            answer = PrecedentAnswer.model_validate(msg["answer"])
            if not answer.refused:
                return answer
    return None


def _try_answer_as_followup(client, situation: str, last_answer: PrecedentAnswer) -> Optional[str]:
    """Returns the follow-up response text, or None if this message isn't
    a follow-up (or answering as one failed) — callers fall through to the
    normal full-retrieval pipeline in either case."""
    with connection(DEFAULT_DB_PATH) as conn:
        app_config = load_app_config()
        budget = NarrationBudgetTracker(conn, cap_usd=app_config.narration.budget_usd)
        with st.spinner("Checking your previous result..."):
            intent = classify_message(client, budget, situation, last_answer)
        if intent != "FOLLOWUP":
            return None
        try:
            return answer_followup(client, budget, situation, last_answer)
        except (NarrationBudgetExceeded, NarrationFailed):
            return None  # fall through to a fresh search rather than fail outright


def _run_new_situation_pipeline(client, situation: str) -> None:
    with st.spinner("Retrieving comparable judgments..."):
        embedder, index, reranker = load_resources(CATEGORY)
        with connection(DEFAULT_DB_PATH) as conn:
            answer = observed_generate_answer(
                conn, index, embedder, CATEGORY, situation, reranker=reranker,
            )

            narrated = None
            if not answer.refused and client is not None:
                try:
                    app_config = load_app_config()
                    budget = NarrationBudgetTracker(conn, cap_usd=app_config.narration.budget_usd)
                    narrated = narrate_answer(client, budget, answer)
                except (NarrationBudgetExceeded, NarrationFailed):
                    narrated = None  # fall back to a deterministic headline below

    if answer.refused:
        headline = ""  # _render_answer shows the warning box using refusal_reason
    elif narrated is not None:
        headline = narrated
    else:
        headline = f"Based on **{answer.sample_size} comparable judgments**, here's a compensation comparison for similar cases:"

    _render_answer(headline, answer.model_dump())
    st.session_state.messages.append({"role": "assistant", "content": headline, "answer": answer.model_dump()})


def _process_situation(situation: str) -> None:
    st.session_state.messages.append({"role": "user", "content": situation, "answer": None})
    with st.chat_message("user"):
        st.markdown(situation)

    with st.chat_message("assistant"):
        client = load_anthropic_client()
        last_answer = _last_successful_answer()

        if client is not None and last_answer is not None:
            followup_text = _try_answer_as_followup(client, situation, last_answer)
            if followup_text is not None:
                st.markdown(followup_text)
                st.session_state.messages.append({"role": "assistant", "content": followup_text, "answer": None})
                return

        _run_new_situation_pipeline(client, situation)


st.title("⚖️ Consumer Court Precedent Finder")
st.caption(
    "Grounded in real NCDRC judgments. Describe your situation — e.g. a builder delaying possession — "
    "and get a compensation comparison with citations, not a generic answer. "
    "Refuses rather than guesses when there isn't enough precedent."
)

with st.sidebar:
    st.subheader("About this data")
    st.markdown(
        "**225 NCDRC judgments** on builder/real-estate possession delay, "
        "structured and independently validated for accuracy."
    )
    st.caption(
        "Every number is computed directly from structured judgment data, not generated — "
        "an LLM only rewrites the already-computed result as prose. If fewer than 10 "
        "genuinely comparable judgments are found, the app refuses rather than guesses."
    )
    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_situation" not in st.session_state:
    st.session_state.pending_situation = None

if not st.session_state.messages:
    st.markdown("#### Try an example, or describe your own situation below")
    cols = st.columns(len(EXAMPLE_SITUATIONS))
    for col, example in zip(cols, EXAMPLE_SITUATIONS):
        if col.button(example, use_container_width=True):
            st.session_state.pending_situation = example

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            _render_answer(msg["content"], msg["answer"])

situation = st.chat_input("Describe your situation...")
if st.session_state.pending_situation:
    situation = st.session_state.pending_situation
    st.session_state.pending_situation = None

if situation:
    _process_situation(situation)
