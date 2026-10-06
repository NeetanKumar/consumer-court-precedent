"""Optional LLM narration of an already-computed PrecedentAnswer into
conversational prose, for the chat UI.

Critical design constraint: the LLM here NEVER computes, estimates, or
invents a number. generate_answer() already computed every statistic
deterministically from the DB (see aggregate.py) before this module ever
runs. This module's only job is to restate that already-verified data in
readable prose — preserving the "every number traces to a real judgment"
guarantee that's the whole point of this project, while fixing the UX
problem of dumping a raw markdown table into a chat window.

All amounts are pre-formatted into Indian digit-grouping strings in
Python (format_inr) BEFORE being sent to the model — the model is told to
copy these strings verbatim, never to reformat a raw number itself. This
isn't a style preference: a real test run had the model asked to "restate
in Indian formatting" and it mis-grouped digits, turning ₹23,80,00,000
into ₹2,38,00,00,000 (10x too large) while following every other
instruction correctly. Indian-style digit grouping (2-digit groups after
the first 3) is exactly the kind of arithmetic LLMs get wrong under
prose-generation pressure — so it's removed from the model's job entirely
rather than trusted to do it right most of the time.

Uses a separate budget ledger (narration_spend_log) from Stage 2's
llm_spend_log — see db.py and config.yaml's `narration` section for why.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Iterator, Optional

import anthropic

from ccpf.answer.schema import ComponentStats, PrecedentAnswer
from ccpf.extract.cost import MICROUSD_PER_USD, estimate_cost_microusd

NARRATION_MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You are presenting a precedent-comparison result to a consumer asking about NCDRC "
    "(Indian consumer court) compensation for their situation. You are given ALREADY-COMPUTED, "
    "verified statistics and cited judgments as structured JSON — a separate system retrieved "
    "these judgments and computed these numbers directly from court records. Your only job is to "
    "present this data conversationally and clearly. Rules:\n"
    "- Every amount and percentage in the JSON is a STRING, already correctly formatted for "
    "display (Indian digit grouping for currency, e.g. '₹23,80,00,000'; percentages like '9.0%'). "
    "Copy these strings EXACTLY, character for character. Do NOT recompute, re-group digits, "
    "round, convert, or reformat them in any way — not even to 'clean up' the formatting.\n"
    "- Do not invent, estimate, or compute any figure not present in the JSON.\n"
    "- If a field's coverage is low (well under half the sample size), say so briefly so the user "
    "knows that figure is based on limited data — don't present it with the same confidence as a "
    "well-covered field.\n"
    "- Reference judgments by their case facts in the prose, not by tid number; end with a short "
    "'Sources' list of tids for anyone who wants to look them up. When you name a specific "
    "judgment as an example, state what was actually AWARDED in that case (its "
    "'relief_awarded_in_this_case' field) rather than only what was claimed — a claimed amount is "
    "not what the consumer received.\n"
    "- Be concise: 150-250 words. No legal advice, no disclaimer paragraph beyond one brief caveat "
    "sentence if coverage is genuinely low."
)


class NarrationBudgetExceeded(RuntimeError):
    pass


class NarrationFailed(RuntimeError):
    pass


def format_inr(amount: float) -> str:
    """Indian digit-grouping: the last 3 digits, then groups of 2. E.g.
    238000000 -> '23,80,00,000' (23.8 crore), not the Western '238,000,000'
    a naive f-string comma-format would produce."""
    n = round(amount)
    sign = "-" if n < 0 else ""
    digits = str(abs(n))
    if len(digits) <= 3:
        return f"{sign}₹{digits}"
    last3, rest = digits[-3:], digits[:-3]
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return f"{sign}₹{','.join(groups)},{last3}"


def _format_stat_value(field_name: str, value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    if field_name == "interest_rate":
        return f"{value:.1f}%"
    return format_inr(value)


def _stats_payload(stats: Optional[ComponentStats]) -> Optional[dict]:
    if stats is None:
        return None
    return {
        "field": stats.field_name,
        "coverage": f"{stats.coverage} of {stats.sample_size} judgments ({stats.coverage_fraction*100:.0f}%)",
        "median": _format_stat_value(stats.field_name, stats.median),
        "min": _format_stat_value(stats.field_name, stats.min),
        "max": _format_stat_value(stats.field_name, stats.max),
    }


def _citation_relief_payload(relief_components: dict) -> dict:
    """The relief actually AWARDED in this one judgment — distinct from
    the aggregated median/min/max in relief_components above, which blend
    across every cited judgment. Without this, the model could only talk
    about what a case claimed, never what it was actually awarded."""
    return {
        "refund": format_inr(relief_components["refund"]) if relief_components.get("refund") is not None else None,
        "interest_rate": (
            f"{relief_components['interest_rate']:.1f}%" if relief_components.get("interest_rate") is not None else None
        ),
        "mental_agony_compensation": (
            format_inr(relief_components["mental_agony_compensation"])
            if relief_components.get("mental_agony_compensation") is not None else None
        ),
        "litigation_cost": (
            format_inr(relief_components["litigation_cost"]) if relief_components.get("litigation_cost") is not None else None
        ),
    }


def build_narration_payload(answer: PrecedentAnswer) -> dict:
    """All numeric fields are converted to pre-formatted display strings
    here — see module docstring for why the LLM never does this itself."""
    return {
        "situation_described": answer.query,
        "sample_size": answer.sample_size,
        "outcome_distribution": answer.outcome_distribution,
        "judgments_that_awarded_money": (
            f"{answer.award_made_count} of {answer.award_known_count}"
            if answer.award_known_count else None
        ),
        "amount_claimed": _stats_payload(answer.amount_claimed_stats),
        "relief_components": {k: _stats_payload(v) for k, v in answer.relief_component_stats.items()},
        "cited_judgments": [
            {
                "tid": c.tid,
                "outcome": c.outcome,
                "amount_claimed": format_inr(c.amount_claimed) if c.amount_claimed is not None else None,
                "relief_awarded_in_this_case": _citation_relief_payload(c.relief_components),
                "fact_summary": c.fact_summary,
            }
            for c in answer.citations
        ],
    }


class NarrationBudgetTracker:
    """Mirrors extract/cost.py's LLMBudgetTracker but against the separate
    narration_spend_log ledger — see that module's docstring for the
    integer-microusd rationale."""

    def __init__(self, conn: sqlite3.Connection, cap_usd: float):
        from ccpf.db import total_narration_spend_microusd

        self._conn = conn
        self.cap_microusd = round(cap_usd * MICROUSD_PER_USD)
        self._spent_microusd = total_narration_spend_microusd(conn)

    @property
    def spent_microusd(self) -> int:
        return self._spent_microusd

    def check(self) -> None:
        if self._spent_microusd >= self.cap_microusd:
            raise NarrationBudgetExceeded(
                f"Narration budget cap reached: spent ${self._spent_microusd / MICROUSD_PER_USD:.4f} "
                f"of ${self.cap_microusd / MICROUSD_PER_USD:.4f} cap."
            )

    def record(self, model: str, ref: str, input_tokens: int, output_tokens: int) -> int:
        cost = estimate_cost_microusd(model, input_tokens, output_tokens)
        self._conn.execute(
            "INSERT INTO narration_spend_log (ts, model, ref, input_tokens, output_tokens, cost_microusd) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), model, ref, input_tokens, output_tokens, cost),
        )
        self._conn.commit()
        self._spent_microusd += cost
        return cost


def narrate_answer(
    client: anthropic.Anthropic,
    budget: NarrationBudgetTracker,
    answer: PrecedentAnswer,
) -> str:
    """Raises NarrationBudgetExceeded / NarrationFailed on any problem —
    callers should catch these and fall back to a raw (non-narrated)
    display rather than let a chat response fail outright."""
    if answer.refused:
        return f"**Insufficient precedent.** {answer.refusal_reason}"

    budget.check()
    payload = json.dumps(build_narration_payload(answer), indent=2, ensure_ascii=False)
    try:
        response = client.messages.create(
            model=NARRATION_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Structured precedent data:\n\n{payload}"}],
        )
    except anthropic.APIError as exc:
        raise NarrationFailed(f"narration API call failed: {exc}") from exc

    usage = response.usage
    # Recorded regardless of what's in the response — same discipline as
    # extract/extractor.py: the call was billed whether or not it's usable.
    budget.record(
        NARRATION_MODEL, ref=answer.query[:80],
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
    )

    if response.stop_reason == "refusal":
        raise NarrationFailed("narration model refused")

    text = "".join(block.text for block in response.content if block.type == "text")
    if not text.strip():
        raise NarrationFailed("narration model returned no text")
    return text


def stream_text(
    client: anthropic.Anthropic,
    budget: NarrationBudgetTracker,
    *,
    model: str,
    system: str,
    user_content: str,
    max_tokens: int,
    ref: str,
) -> Iterator[str]:
    """Yield the model's text as it is generated, for st.write_stream.

    Spend is recorded once, from the final message's usage, in a `finally`
    so a consumer that stops reading early (user hits Stop) still pays only
    for what was billed and the ledger stays honest. Raises
    NarrationBudgetExceeded before any call, and NarrationFailed on API
    errors, refusals or empty output — callers fall back exactly as with
    the non-streaming path.
    """
    budget.check()
    emitted = False
    try:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        ) as stream:
            final = None
            try:
                for chunk in stream.text_stream:
                    if chunk:
                        emitted = True
                        yield chunk
            finally:
                # A broken stream has no final message to read usage from;
                # don't let that mask the original error.
                try:
                    final = stream.get_final_message()
                    budget.record(
                        model, ref=ref,
                        input_tokens=final.usage.input_tokens, output_tokens=final.usage.output_tokens,
                    )
                except Exception:
                    final = None
            if final is not None and final.stop_reason == "refusal":
                raise NarrationFailed("model refused")
    except anthropic.APIError as exc:
        raise NarrationFailed(f"streaming API call failed: {exc}") from exc
    if not emitted:
        raise NarrationFailed("model returned no text")


def narrate_answer_stream(
    client: anthropic.Anthropic, budget: NarrationBudgetTracker, answer: PrecedentAnswer
) -> Iterator[str]:
    payload = json.dumps(build_narration_payload(answer), indent=2, ensure_ascii=False)
    return stream_text(
        client, budget, model=NARRATION_MODEL, system=SYSTEM_PROMPT,
        user_content=f"Structured precedent data:\n\n{payload}",
        max_tokens=1024, ref=answer.query[:80],
    )
