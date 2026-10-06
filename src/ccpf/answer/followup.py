"""Conversational follow-ups: without this, every chat message re-runs the
full retrieval pipeline from scratch treating it as an independent new
situation — so "what is the median interest rate" right after a real
answer gets searched for as if IT were a consumer dispute, finds almost
nothing, and falsely refuses.

Two-category classification only (kept deliberately minimal, not a full
intent taxonomy): a new message is either
  NEW      — a different/new situation, needs fresh retrieval, or
  FOLLOWUP — a question about the data already shown in the previous
             answer, answered from that existing data with no new
             retrieval call at all.

Both the classifier and the follow-up answerer are cheap Haiku calls and
share narrate.py's NarrationBudgetTracker/ledger — this is the same
category of live per-query chat spend, not a reason to add a third ledger.
Classification fails open to NEW on any error: mistaking a genuine new
situation for a follow-up (and answering from stale data) is worse than
the reverse (an unnecessary but harmless fresh search).
"""
from __future__ import annotations

import json

import anthropic

from ccpf.answer.narrate import NarrationBudgetTracker, NarrationFailed, build_narration_payload, stream_text
from ccpf.answer.schema import PrecedentAnswer

FOLLOWUP_MODEL = "claude-haiku-4-5"

CLASSIFY_SYSTEM_PROMPT = (
    "You classify a user's chat message in a legal-precedent-lookup assistant. The assistant's "
    "PREVIOUS answer (compensation statistics for a consumer dispute situation) is given below. "
    "Decide whether the user's NEW message is:\n"
    "NEW — describes a new or different situation that needs a fresh search.\n"
    "FOLLOWUP — a question about the numbers, cases, or data already shown in the previous answer "
    "(e.g. 'what was the interest rate', 'tell me more about that case', 'why is coverage low').\n"
    "Respond with EXACTLY one word: NEW or FOLLOWUP."
)

FOLLOWUP_SYSTEM_PROMPT = (
    "You are answering a follow-up question about a precedent-comparison result you already gave "
    "the user, using ONLY the structured JSON data below — already computed and verified. Do not "
    "invent, estimate, or compute any number not present in it. If the question asks about "
    "something this data doesn't cover, say so plainly and suggest describing a new situation "
    "instead of guessing.\n"
    "- Every amount/percentage in the JSON is a pre-formatted STRING. Copy it EXACTLY, unchanged.\n"
    "- Be concise: 1-3 sentences is enough for a simple factual question."
)


def classify_message(
    client: anthropic.Anthropic,
    budget: NarrationBudgetTracker,
    message: str,
    last_answer: PrecedentAnswer,
) -> str:
    """Returns "NEW" or "FOLLOWUP"."""
    try:
        budget.check()
        payload = json.dumps(build_narration_payload(last_answer), indent=2, ensure_ascii=False)
        response = client.messages.create(
            model=FOLLOWUP_MODEL,
            max_tokens=10,
            system=CLASSIFY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Previous answer data:\n{payload}\n\nNew message: {message}"}],
        )
        usage = response.usage
        budget.record(
            FOLLOWUP_MODEL, ref=f"classify:{message[:60]}",
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip().upper()
        return "FOLLOWUP" if "FOLLOWUP" in text else "NEW"
    except Exception:
        return "NEW"


def answer_followup(
    client: anthropic.Anthropic,
    budget: NarrationBudgetTracker,
    message: str,
    last_answer: PrecedentAnswer,
) -> str:
    """Raises NarrationFailed on any problem — callers should catch this
    and fall back to running the full new-situation pipeline instead."""
    budget.check()
    payload = json.dumps(build_narration_payload(last_answer), indent=2, ensure_ascii=False)
    try:
        response = client.messages.create(
            model=FOLLOWUP_MODEL,
            max_tokens=512,
            system=FOLLOWUP_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": f"Previous answer data:\n{payload}\n\nFollow-up question: {message}"}],
        )
    except anthropic.APIError as exc:
        raise NarrationFailed(f"followup API call failed: {exc}") from exc

    usage = response.usage
    budget.record(
        FOLLOWUP_MODEL, ref=f"followup:{message[:60]}",
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
    )
    if response.stop_reason == "refusal":
        raise NarrationFailed("followup model refused")
    text = "".join(b.text for b in response.content if b.type == "text")
    if not text.strip():
        raise NarrationFailed("followup model returned no text")
    return text


def answer_followup_stream(
    client: anthropic.Anthropic,
    budget: NarrationBudgetTracker,
    message: str,
    last_answer: PrecedentAnswer,
):
    payload = json.dumps(build_narration_payload(last_answer), indent=2, ensure_ascii=False)
    return stream_text(
        client, budget, model=FOLLOWUP_MODEL, system=FOLLOWUP_SYSTEM_PROMPT,
        user_content=f"Previous answer data:\n{payload}\n\nFollow-up question: {message}",
        max_tokens=512, ref=f"followup:{message[:60]}",
    )
