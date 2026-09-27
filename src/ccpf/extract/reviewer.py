"""Independent gold-label reviewer for the Stage 2 validation set.

Deliberately a SEPARATE code path from extractor.py, using a different
model family (Claude Fable 5, vs. the Haiku/Sonnet pair doing extraction)
and a prompt that never sees the extraction output — the point is an
accuracy signal uncorrelated with whatever mistakes the extraction
pipeline itself makes.
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

REVIEWER_MODEL = "claude-fable-5"
MAX_DOC_CHARS = 40_000

REVIEWER_SYSTEM_PROMPT = (
    "You are an independent reviewer extracting structured facts from an Indian "
    "consumer court (NCDRC) judgment about a builder/real-estate possession delay "
    "dispute, for use as ground truth to evaluate a separate extraction pipeline. "
    "Extract only what the judgment text actually states — do not infer amounts "
    "or outcomes that are not explicitly present. If a field cannot be determined "
    "from the text, leave it null rather than guessing. Read carefully: this label "
    "is the standard another system's output will be graded against."
)


@dataclass
class ReviewResult:
    judgment: Judgment
    input_tokens: int
    output_tokens: int


class ReviewFailed(RuntimeError):
    pass


class IndependentReviewer:
    """Model-agnostic wrapper — currently configured for Claude Fable 5, but
    any model can be passed in via `model=`, e.g. for A/B-ing reviewers."""

    def __init__(self, client: anthropic.Anthropic, budget: LLMBudgetTracker, model: str = REVIEWER_MODEL):
        self._client = client
        self._budget = budget
        self.model = model

    def review(self, tid: int, plain_text: str) -> ReviewResult:
        self._budget.check()
        text = plain_text[:MAX_DOC_CHARS]
        try:
            response = self._client.messages.parse(
                model=self.model,
                max_tokens=4096,
                system=REVIEWER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"Judgment text:\n\n{text}"}],
                output_format=Judgment,
            )
        except pydantic.ValidationError as exc:
            # Same failure mode as extractor.py._call — the SDK validates
            # inside parse() itself, so there's no usage to record here.
            raise ReviewFailed(f"tid={tid}: {self.model} produced unparseable output: {exc}") from exc
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        # Recorded regardless of parse success — Anthropic bills on the
        # call, matching the same discipline as extract/extractor.py.
        self._budget.record(self.model, ref=str(tid), input_tokens=input_tokens, output_tokens=output_tokens)

        if response.stop_reason == "refusal":
            raise ReviewFailed(f"tid={tid}: {self.model} refused review")
        if response.parsed_output is None:
            raise ReviewFailed(f"tid={tid}: {self.model} produced no parsed output (stop_reason={response.stop_reason})")

        return ReviewResult(judgment=response.parsed_output, input_tokens=input_tokens, output_tokens=output_tokens)
