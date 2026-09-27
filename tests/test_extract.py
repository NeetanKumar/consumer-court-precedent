"""Stage 2 extraction tests. No real Anthropic calls — a fake client stands
in for anthropic.Anthropic, returning canned messages.parse() responses.
"""
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from ccpf.extract.cost import LLMBudgetExceeded, LLMBudgetTracker, estimate_cost_microusd
from ccpf.extract.extractor import (
    HAIKU_MODEL,
    SONNET_MODEL,
    ClaudeExtractor,
    ExtractionFailed,
)
from ccpf.extract.run import run_extraction
from ccpf.extract.schema import Judgment, ReliefComponents


def _judgment(confidence="high", outcome="allowed"):
    return Judgment(
        case_type="builder possession delay",
        fact_summary="Complainant booked a flat; possession delayed by 3 years.",
        dispute_duration_months=36,
        amount_claimed=1500000,
        relief_components=ReliefComponents(refund=1500000, interest_rate=9.0),
        outcome=outcome,
        forum_level="NCDRC",
        confidence=confidence,
    )


def _response(parsed_output, stop_reason="end_turn", input_tokens=1000, output_tokens=200):
    return SimpleNamespace(
        parsed_output=parsed_output,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeAnthropicClient:
    """Stands in for anthropic.Anthropic — queues canned parse() responses per model."""

    def __init__(self, responses_by_model: dict[str, list]):
        self._responses = {k: list(v) for k, v in responses_by_model.items()}
        self.calls = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, model, **kwargs):
        self.calls.append(model)
        queue = self._responses[model]
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_cost_estimate_matches_pricing_table():
    # Haiku: $1/$5 per MTok
    cost = estimate_cost_microusd(HAIKU_MODEL, input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 6_000_000  # $1 + $5 = $6, in microusd


def test_budget_tracker_survives_reconstruction(db_conn):
    budget1 = LLMBudgetTracker(db_conn, cap_usd=10.0)
    budget1.record(HAIKU_MODEL, ref="1", input_tokens=1000, output_tokens=200)
    spent_after_first = budget1.spent_microusd
    assert spent_after_first > 0

    budget2 = LLMBudgetTracker(db_conn, cap_usd=10.0)
    assert budget2.spent_microusd == spent_after_first


def test_budget_exceeded_stops_further_calls(db_conn):
    budget = LLMBudgetTracker(db_conn, cap_usd=0.0)  # zero cap: any spend exceeds it
    with pytest.raises(LLMBudgetExceeded):
        budget.check()


def test_high_confidence_uses_primary_model_only(db_conn):
    client = FakeAnthropicClient({HAIKU_MODEL: [_response(_judgment(confidence="high"))]})
    budget = LLMBudgetTracker(db_conn, cap_usd=10.0)
    extractor = ClaudeExtractor(client, budget)

    result = extractor.extract(tid=1, plain_text="some judgment text")

    assert result.model_used == HAIKU_MODEL
    assert result.escalated is False
    assert client.calls == [HAIKU_MODEL]


def test_low_confidence_escalates_to_sonnet(db_conn):
    client = FakeAnthropicClient({
        HAIKU_MODEL: [_response(_judgment(confidence="low"))],
        SONNET_MODEL: [_response(_judgment(confidence="high"))],
    })
    budget = LLMBudgetTracker(db_conn, cap_usd=10.0)
    extractor = ClaudeExtractor(client, budget)

    result = extractor.extract(tid=2, plain_text="ambiguous text")

    assert result.model_used == SONNET_MODEL
    assert result.escalated is True
    assert client.calls == [HAIKU_MODEL, SONNET_MODEL]


def test_refused_primary_escalates(db_conn):
    client = FakeAnthropicClient({
        HAIKU_MODEL: [_response(None, stop_reason="refusal")],
        SONNET_MODEL: [_response(_judgment(confidence="high"))],
    })
    budget = LLMBudgetTracker(db_conn, cap_usd=10.0)
    extractor = ClaudeExtractor(client, budget)

    result = extractor.extract(tid=3, plain_text="text")

    assert result.model_used == SONNET_MODEL
    # Spend recorded for the refused call too — Anthropic bills on the call.
    assert budget.spent_microusd > 0


def test_truncated_json_output_escalates_gracefully(db_conn):
    """Regression test: a real run hit this exact failure — Haiku's output
    got truncated mid-JSON (hit max_tokens before finishing), and the SDK's
    client.messages.parse() raises pydantic.ValidationError from inside the
    call itself, before any response object is returned. That must be
    treated as a failed extraction (triggering escalation), not crash the
    whole run with an unhandled exception."""
    import pydantic
    try:
        Judgment.model_validate_json('{"case_type": "truncated mid-str')
        assert False, "expected this to raise"
    except pydantic.ValidationError as exc:
        parse_error = exc

    client = FakeAnthropicClient({
        HAIKU_MODEL: [parse_error],
        SONNET_MODEL: [_response(_judgment(confidence="high"))],
    })
    budget = LLMBudgetTracker(db_conn, cap_usd=10.0)
    extractor = ClaudeExtractor(client, budget)

    result = extractor.extract(tid=99, plain_text="text")

    assert result.model_used == SONNET_MODEL
    assert result.escalated is True
    assert client.calls == [HAIKU_MODEL, SONNET_MODEL]


def test_both_models_fail_raises(db_conn):
    client = FakeAnthropicClient({
        HAIKU_MODEL: [_response(None, stop_reason="refusal")],
        SONNET_MODEL: [_response(None, stop_reason="refusal")],
    })
    budget = LLMBudgetTracker(db_conn, cap_usd=10.0)
    extractor = ClaudeExtractor(client, budget)

    with pytest.raises(ExtractionFailed):
        extractor.extract(tid=4, plain_text="text")


def test_run_extraction_is_idempotent(db_conn):
    now = "2026-01-01T00:00:00+00:00"
    db_conn.execute(
        "INSERT INTO raw_docs (tid, title, docsource, publishdate, raw_html, plain_text, fetched_at, cost_paise) "
        "VALUES (10, 't', 'National Consumer Disputes Redressal Commission', '2021-01-01', '', 'judgment text', ?, 20)",
        (now,),
    )
    db_conn.execute(
        "INSERT INTO doc_filters (tid, category, is_ncdrc, has_award_language, matched_keywords, passed, filtered_at) "
        "VALUES (10, 'builder_delay', 1, 1, '', 1, ?)",
        (now,),
    )
    db_conn.commit()

    client = FakeAnthropicClient({HAIKU_MODEL: [_response(_judgment(confidence="high"))]})
    budget = LLMBudgetTracker(db_conn, cap_usd=10.0)
    extractor = ClaudeExtractor(client, budget)

    summary1 = run_extraction(extractor, db_conn, budget, "builder_delay")
    assert summary1.succeeded == 1
    assert client.calls == [HAIKU_MODEL]

    # Second run must not re-call the LLM for the already-extracted doc.
    summary2 = run_extraction(extractor, db_conn, budget, "builder_delay")
    assert summary2.attempted == 0
    assert summary2.succeeded == 0
    assert client.calls == [HAIKU_MODEL]


def test_force_reextraction_with_tids(db_conn):
    """--tids + --force is how a prompt change gets tested cheaply on a known
    subset before a full re-run: it must delete and re-extract only the
    named docs, and leave everything else untouched."""
    now = "2026-01-01T00:00:00+00:00"
    for tid in (30, 31):
        db_conn.execute(
            "INSERT INTO raw_docs (tid, title, docsource, publishdate, raw_html, plain_text, fetched_at, cost_paise) "
            "VALUES (?, 't', 'National Consumer Disputes Redressal Commission', '2021-01-01', '', 'text', ?, 20)",
            (tid, now),
        )
        db_conn.execute(
            "INSERT INTO doc_filters (tid, category, is_ncdrc, has_award_language, matched_keywords, passed, filtered_at) "
            "VALUES (?, 'builder_delay', 1, 1, '', 1, ?)",
            (tid, now),
        )
    db_conn.commit()

    client = FakeAnthropicClient({
        HAIKU_MODEL: [_response(_judgment(confidence="high")), _response(_judgment(confidence="high"))]
    })
    budget = LLMBudgetTracker(db_conn, cap_usd=10.0)
    extractor = ClaudeExtractor(client, budget)

    run_extraction(extractor, db_conn, budget, "builder_delay")  # extracts both 30 and 31
    assert client.calls == [HAIKU_MODEL, HAIKU_MODEL]

    # Re-run restricted to tid 30 with --force: must re-call the LLM for 30
    # only, leaving 31's existing row untouched.
    client2 = FakeAnthropicClient({HAIKU_MODEL: [_response(_judgment(confidence="medium"))]})
    extractor2 = ClaudeExtractor(client2, budget)
    summary = run_extraction(extractor2, db_conn, budget, "builder_delay", tids=[30], force=True)

    assert summary.succeeded == 1
    assert client2.calls == [HAIKU_MODEL]

    row30 = db_conn.execute("SELECT confidence FROM extractions WHERE tid = 30").fetchone()
    row31 = db_conn.execute("SELECT confidence FROM extractions WHERE tid = 31").fetchone()
    assert row30["confidence"] == "medium"  # overwritten by the forced re-extraction
    assert row31["confidence"] == "high"  # untouched


def test_force_failure_preserves_existing_row(db_conn):
    """Regression test: a real run deleted 40 docs' extraction rows upfront
    (on --force), then hit an API error before any of them re-extracted
    successfully — leaving those 40 docs with NO extraction row at all
    until the next successful run. The fix must overwrite atomically
    (UPSERT on success), never delete-then-hope."""
    now = "2026-01-01T00:00:00+00:00"
    db_conn.execute(
        "INSERT INTO raw_docs (tid, title, docsource, publishdate, raw_html, plain_text, fetched_at, cost_paise) "
        "VALUES (40, 't', 'National Consumer Disputes Redressal Commission', '2021-01-01', '', 'text', ?, 20)",
        (now,),
    )
    db_conn.execute(
        "INSERT INTO doc_filters (tid, category, is_ncdrc, has_award_language, matched_keywords, passed, filtered_at) "
        "VALUES (40, 'builder_delay', 1, 1, '', 1, ?)",
        (now,),
    )
    db_conn.execute(
        "INSERT INTO extractions (tid, category, model_used, escalated, confidence, judgment_json, extracted_at) "
        "VALUES (40, 'builder_delay', 'claude-haiku-4-5', 0, 'medium', ?, ?)",
        (_judgment(confidence="medium").model_dump_json(), now),
    )
    db_conn.commit()

    # Budget already exhausted — the re-extraction attempt fails immediately.
    budget = LLMBudgetTracker(db_conn, cap_usd=0.0)
    client = FakeAnthropicClient({HAIKU_MODEL: []})
    extractor = ClaudeExtractor(client, budget)

    summary = run_extraction(extractor, db_conn, budget, "builder_delay", tids=[40], force=True)

    assert summary.stopped_early is not None
    row = db_conn.execute("SELECT confidence FROM extractions WHERE tid = 40").fetchone()
    assert row is not None  # NOT deleted
    assert row["confidence"] == "medium"  # untouched — old value preserved


def test_account_level_api_error_stops_run_cleanly(db_conn):
    """Regression test: an account-level usage-limit error (or any Anthropic
    API error) must stop the run with a clear message and preserve already-
    extracted rows, not crash with a raw traceback. This is exactly what
    happened on the real Rs.10 run: 122/225 docs succeeded, then the account
    hit its usage cap and the script crashed instead of stopping gracefully."""
    now = "2026-01-01T00:00:00+00:00"
    for tid in (20, 21):
        db_conn.execute(
            "INSERT INTO raw_docs (tid, title, docsource, publishdate, raw_html, plain_text, fetched_at, cost_paise) "
            "VALUES (?, 't', 'National Consumer Disputes Redressal Commission', '2021-01-01', '', 'judgment text', ?, 20)",
            (tid, now),
        )
        db_conn.execute(
            "INSERT INTO doc_filters (tid, category, is_ncdrc, has_award_language, matched_keywords, passed, filtered_at) "
            "VALUES (?, 'builder_delay', 1, 1, '', 1, ?)",
            (tid, now),
        )
    db_conn.commit()

    api_error = anthropic.BadRequestError(
        "usage limits reached",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")),
        body={"error": {"type": "invalid_request_error", "message": "usage limits reached"}},
    )
    client = FakeAnthropicClient({
        HAIKU_MODEL: [_response(_judgment(confidence="high")), api_error],
    })
    budget = LLMBudgetTracker(db_conn, cap_usd=10.0)
    extractor = ClaudeExtractor(client, budget)

    summary = run_extraction(extractor, db_conn, budget, "builder_delay")

    assert summary.succeeded == 1  # first doc succeeded and is preserved
    assert summary.stopped_early is not None
    assert "usage limits" in summary.stopped_early

    row = db_conn.execute("SELECT COUNT(*) AS n FROM extractions").fetchone()
    assert row["n"] == 1
