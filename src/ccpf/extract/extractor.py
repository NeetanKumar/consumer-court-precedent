"""Tiered structured extraction: Haiku 4.5 by default, escalate to Sonnet 5
on low self-reported confidence or a failed/refused Haiku attempt.

Rationale for tiering (validated against the hand-labeled set in Stage 2,
not assumed): Haiku is cheap and fast enough to run schema-filling over the
whole corpus; escalating only the ambiguous documents keeps spend down
without trading away accuracy where it matters.

Uses client.messages.parse() (Structured Outputs) rather than free-text
+ manual JSON parsing — it validates the response against the Judgment
pydantic model directly, which is the documented, recommended approach for
this exact use case (classification/extraction/Q&A) per the Claude API
skill.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import anthropic
import pydantic

from ccpf.extract.cost import LLMBudgetTracker
from ccpf.extract.schema import Judgment

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5"
SONNET_MODEL = "claude-sonnet-5"

MAX_DOC_CHARS = 40_000  # keep prompts well inside Haiku's 200K context, cheaply

SYSTEM_PROMPT = (
    "You extract structured facts from Indian consumer court (NCDRC) judgments "
    "about builder/real-estate possession delay disputes. Extract only what the "
    "judgment text actually states — do not infer amounts or outcomes that are "
    "not explicitly present. If a field cannot be determined from the text, "
    "leave it null rather than guessing.\n\n"
    "relief_components (refund, interest_rate, mental_agony_compensation, "
    "litigation_cost) must reflect what the Commission actually AWARDED in its "
    "final operative order/directions — usually near the end of the judgment "
    "(e.g. 'the opposite party is directed to...', 'it is ordered that...'). "
    "Do NOT use:\n"
    "- amounts or rates from the complainant's prayer/relief sought section — "
    "that is what was CLAIMED, not what was awarded, and may differ from the "
    "final order;\n"
    "- generic rates cited in the legal reasoning as precedent or a standard "
    "benchmark (NCDRC judgments very commonly cite '9% per annum' as a general "
    "reference point in their reasoning, distinct from the case-specific rate "
    "actually awarded here) — only the rate/amount in the operative order counts;\n"
    "- amounts mentioned only in the facts/background section describing what "
    "the complainant originally paid or sought.\n"
    "If the operative order does not state a component explicitly, leave it "
    "null even if a related number appears elsewhere in the judgment."
)


@dataclass
class ExtractionResult:
    judgment: Judgment
    model_used: str
    escalated: bool
    input_tokens: int
    output_tokens: int


class ExtractionFailed(RuntimeError):
    pass


class Extractor:
    """Protocol-ish base: any extractor must implement `extract`."""

    def extract(self, tid: int, plain_text: str) -> ExtractionResult:
        raise NotImplementedError


class ClaudeExtractor(Extractor):
    def __init__(
        self,
        client: anthropic.Anthropic,
        budget: LLMBudgetTracker,
        primary_model: str = HAIKU_MODEL,
        escalation_model: str = SONNET_MODEL,
    ):
        self._client = client
        self._budget = budget
        self._primary_model = primary_model
        self._escalation_model = escalation_model

    def extract(self, tid: int, plain_text: str) -> ExtractionResult:
        self._budget.check()
        judgment, usage = self._call(self._primary_model, tid, plain_text)

        if judgment is None or judgment.confidence == "low":
            self._budget.check()
            escalated_judgment, escalated_usage = self._call(self._escalation_model, tid, plain_text)
            if escalated_judgment is not None:
                return ExtractionResult(
                    judgment=escalated_judgment,
                    model_used=self._escalation_model,
                    escalated=True,
                    input_tokens=escalated_usage[0],
                    output_tokens=escalated_usage[1],
                )
            if judgment is not None:
                # Escalation also failed to parse; fall back to the primary
                # model's result rather than losing the document entirely.
                return ExtractionResult(
                    judgment=judgment,
                    model_used=self._primary_model,
                    escalated=False,
                    input_tokens=usage[0],
                    output_tokens=usage[1],
                )
            raise ExtractionFailed(f"tid={tid}: both {self._primary_model} and {self._escalation_model} failed")

        return ExtractionResult(
            judgment=judgment,
            model_used=self._primary_model,
            escalated=False,
            input_tokens=usage[0],
            output_tokens=usage[1],
        )

    def _call(self, model: str, tid: int, plain_text: str) -> tuple[Optional[Judgment], tuple[int, int]]:
        text = plain_text[:MAX_DOC_CHARS]
        try:
            response = self._client.messages.parse(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Judgment text:\n\n{text}"}],
                output_format=Judgment,
            )
        except pydantic.ValidationError as exc:
            # The SDK validates the model's JSON output against Judgment
            # *inside* client.messages.parse() itself, before returning a
            # response object — so on a truncated/malformed output (e.g. the
            # model hit max_tokens mid-JSON) there is no response to read
            # usage from here. This is a genuine parse failure, same
            # category as a refusal or empty parsed_output; treat it the
            # same way rather than letting it crash the whole run.
            logger.warning("tid=%s: %s produced unparseable output: %s", tid, model, exc)
            return None, (0, 0)
        usage = (response.usage.input_tokens, response.usage.output_tokens)
        # Spend is recorded on every completed call, regardless of whether
        # parsing succeeded — Anthropic bills on the call, not on our
        # downstream validation (same principle as ingest/client.py).
        self._budget.record(model, ref=str(tid), input_tokens=usage[0], output_tokens=usage[1])

        if response.stop_reason == "refusal":
            logger.warning("tid=%s: %s refused extraction", tid, model)
            return None, usage
        if response.parsed_output is None:
            logger.warning("tid=%s: %s produced no parsed output (stop_reason=%s)", tid, model, response.stop_reason)
            return None, usage
        return response.parsed_output, usage
