"""Streamlit rendering for precedent answers: metric cards, a real award
distribution chart, and citation cards. Pure display — no retrieval, no
LLM calls, no state — so a live answer and a replayed one look identical.
"""
from __future__ import annotations

from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

from ccpf.answer.narrate import format_inr
from ccpf.answer.schema import Citation, ComponentStats, PrecedentAnswer
from ccpf.ui.limits import DISCLAIMER

INDIANKANOON_DOC_URL = "https://indiankanoon.org/doc/{tid}/"
LOW_COVERAGE = 0.5
MIN_POINTS_FOR_CHART = 5

OUTCOME_BADGE = {
    "allowed": ("Allowed", "green"),
    "partly_allowed": ("Partly allowed", "orange"),
    "dismissed": ("Dismissed", "red"),
}

FIELD_LABELS = {
    "amount_claimed": "Amount claimed",
    "refund": "Refund",
    "interest_rate": "Interest rate",
    "mental_agony_compensation": "Mental agony compensation",
    "litigation_cost": "Litigation cost",
}


def format_value(field_name: str, value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if field_name == "interest_rate":
        return f"{value:.1f}%"
    return format_inr(value)


def _coverage_text(stats: ComponentStats) -> str:
    return f"based on {stats.coverage} of {stats.sample_size} judgments"


def _metric(col, label: str, stats: ComponentStats) -> None:
    low = stats.coverage_fraction < LOW_COVERAGE
    with col:
        st.metric(
            label, format_value(stats.field_name, stats.median),
            help="Median across the judgments that stated this value. "
            + ("Limited data — treat with caution." if low else ""),
            border=True,
        )
        if low:
            st.caption(f":orange[:material/warning:] Limited data: {_coverage_text(stats)}")
        else:
            st.caption(_coverage_text(stats).capitalize())


def render_metrics(answer: PrecedentAnswer) -> None:
    cols = st.columns(3)
    with cols[0]:
        st.metric("Comparable judgments", answer.sample_size, border=True)
        if answer.duration_window:
            lo, hi = answer.duration_window
            st.caption(f"Delay between {lo} and {hi} months")
        elif answer.facet_relaxed:
            st.caption(f"Few cases with a ~{answer.duration_months_requested}-month delay, so all delay lengths are shown")

    refund = answer.relief_component_stats.get("refund")
    if refund and refund.median is not None:
        _metric(cols[1], "Median refund awarded", refund)
    elif answer.amount_claimed_stats and answer.amount_claimed_stats.median is not None:
        _metric(cols[1], "Median amount claimed", answer.amount_claimed_stats)

    interest = answer.relief_component_stats.get("interest_rate")
    if interest and interest.median is not None:
        _metric(cols[2], "Median interest rate", interest)


def distribution_chart(stats: ComponentStats) -> Optional[alt.Chart]:
    """Strip plot of every contributing value on a log axis (awards span
    orders of magnitude), with the interquartile band and the median."""
    vals = [v for v in stats.values if v and v > 0]
    if len(vals) < MIN_POINTS_FOR_CHART:
        return None
    df = pd.DataFrame({"tid": stats.contributing_tids[: len(stats.values)], "value": stats.values})
    df = df[df["value"] > 0]
    df["amount"] = df["value"].map(lambda v: format_value(stats.field_name, v))
    x = alt.X("value:Q", scale=alt.Scale(type="log"), axis=alt.Axis(title=None, format="~s"))
    points = alt.Chart(df).mark_circle(size=70, opacity=0.55).encode(
        x=x, y=alt.value(30), tooltip=[alt.Tooltip("tid:N", title="Judgment"), alt.Tooltip("amount:N", title="Amount")]
    )
    layers = [points]
    if stats.p25 and stats.p75 and stats.p25 > 0:
        band = pd.DataFrame({"lo": [stats.p25], "hi": [stats.p75]})
        layers.insert(0, alt.Chart(band).mark_rect(opacity=0.15, color="gray").encode(x="lo:Q", x2="hi:Q"))
    if stats.median and stats.median > 0:
        med = pd.DataFrame({"m": [stats.median]})
        layers.append(alt.Chart(med).mark_rule(strokeWidth=2).encode(x="m:Q"))
    return alt.layer(*layers).properties(height=70)


def render_distribution(answer: PrecedentAnswer) -> None:
    refund = answer.relief_component_stats.get("refund")
    chart = distribution_chart(refund) if refund else None
    if chart is None:
        return
    st.caption(
        f"Refund awarded in each judgment ({refund.coverage}); the shaded band is the middle 50% "
        "and the line is the median. Hover a dot for its judgment id."
    )
    st.altair_chart(chart, width="stretch")


def outcome_badges(answer: PrecedentAnswer) -> None:
    parts = []
    for outcome, n in answer.outcome_distribution.items():
        label, color = OUTCOME_BADGE.get(outcome, (outcome, "gray"))
        parts.append(f":{color}-badge[{label}: {n}]")
    if parts:
        st.markdown(" ".join(parts))


def _relief_line(c: Citation) -> str:
    parts = []
    if c.amount_claimed is not None:
        parts.append(f"claimed {format_inr(c.amount_claimed)}")
    r = c.relief_components
    if r.get("refund") is not None:
        parts.append(f"refund awarded {format_inr(r['refund'])}")
    if r.get("interest_rate") is not None:
        parts.append(f"interest {r['interest_rate']:.1f}%")
    if r.get("mental_agony_compensation") is not None:
        parts.append(f"mental agony {format_inr(r['mental_agony_compensation'])}")
    if r.get("litigation_cost") is not None:
        parts.append(f"litigation cost {format_inr(r['litigation_cost'])}")
    return " · ".join(parts) if parts else "amounts not extracted for this judgment"


def render_citation(c: Citation) -> None:
    label, color = OUTCOME_BADGE.get(c.outcome, (c.outcome, "gray"))
    title = c.title or f"Judgment {c.tid}"
    with st.container(border=True):
        st.markdown(f"**[{title}]({INDIANKANOON_DOC_URL.format(tid=c.tid)})**")
        meta = [f":{color}-badge[{label}]"]
        if c.forum_level:
            meta.append(c.forum_level)
        if c.publishdate:
            meta.append(c.publishdate)
        st.markdown(" · ".join(meta))
        st.caption(_relief_line(c))
        summary = c.fact_summary[:300] + ("…" if len(c.fact_summary) > 300 else "")
        st.write(summary)


def render_details(answer: PrecedentAnswer) -> None:
    with st.expander("Statistics and cited judgments", icon=":material/table_chart:"):
        outcome_badges(answer)
        if answer.award_known_count:
            st.caption(
                f"{answer.award_made_count} of {answer.award_known_count} judgments directed a monetary award; "
                "the rest were remands, procedural orders or dismissals."
            )
        stats_list = [answer.amount_claimed_stats] if answer.amount_claimed_stats else []
        stats_list += list(answer.relief_component_stats.values())
        rows = [
            {
                "Field": FIELD_LABELS.get(s.field_name, s.field_name),
                "Coverage": f"{s.coverage}/{s.sample_size} ({s.coverage_fraction * 100:.0f}%)",
                "Median": format_value(s.field_name, s.median),
                "Middle 50%": (
                    f"{format_value(s.field_name, s.p25)} – {format_value(s.field_name, s.p75)}"
                    if s.p25 is not None and s.p75 is not None else "n/a"
                ),
                "Min": format_value(s.field_name, s.min),
                "Max": format_value(s.field_name, s.max),
            }
            for s in stats_list
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown("**Cited judgments**")
        for c in answer.citations:
            render_citation(c)


def render_answer_body(answer: PrecedentAnswer) -> None:
    """Everything under the narrated text for a successful answer."""
    render_metrics(answer)
    render_distribution(answer)
    render_details(answer)
    st.caption(f"{DISCLAIMER} Based on {answer.sample_size} comparable judgments.")


def render_refusal(answer: PrecedentAnswer) -> None:
    st.warning(f"**Not enough precedent to answer reliably.**\n\n{answer.refusal_reason}", icon=":material/info:")
    st.caption(DISCLAIMER)
