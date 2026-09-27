"""Conversational follow-up tests. No real Anthropic calls."""
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from ccpf.answer.followup import answer_followup, classify_message
from ccpf.answer.narrate import NarrationBudgetTracker, NarrationFailed
from ccpf.answer.schema import ComponentStats, PrecedentAnswer


def _answer():
    return PrecedentAnswer(
        query="builder delayed possession",
        filters_applied={},
        sample_size=12,
        refused=False,
        outcome_distribution={"allowed": 10, "dismissed": 2},
        amount_claimed_stats=ComponentStats(
            field_name="amount_claimed", sample_size=12, coverage=10,
            median=1000000, min=500000, max=2000000, contributing_tids=list(range(1, 11)),
        ),
    )


class FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


def _response(text, stop_reason="end_turn", input_tokens=300, output_tokens=20):
    return SimpleNamespace(
        content=[FakeTextBlock(text)] if text else [],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_classify_new_message(db_conn):
    client = FakeClient([_response("NEW")])
    budget = NarrationBudgetTracker(db_conn, cap_usd=5.0)

    result = classify_message(client, budget, "builder in Pune delayed my flat by 4 years", _answer())

    assert result == "NEW"
    assert budget.spent_microusd > 0


def test_classify_followup_message(db_conn):
    client = FakeClient([_response("FOLLOWUP")])
    budget = NarrationBudgetTracker(db_conn, cap_usd=5.0)

    result = classify_message(client, budget, "what is the median interest rate", _answer())

    assert result == "FOLLOWUP"


def test_classify_fails_open_to_new_on_api_error(db_conn):
    api_error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    client = FakeClient([api_error])
    budget = NarrationBudgetTracker(db_conn, cap_usd=5.0)

    result = classify_message(client, budget, "what is the median interest rate", _answer())

    assert result == "NEW"  # never silently treats an error as a confident followup


def test_classify_fails_open_to_new_on_budget_exceeded(db_conn):
    client = FakeClient([_response("FOLLOWUP")])
    budget = NarrationBudgetTracker(db_conn, cap_usd=0.0)  # zero cap

    result = classify_message(client, budget, "what is the median interest rate", _answer())

    assert result == "NEW"
    assert client.calls == 0  # never even attempted the call


def test_answer_followup_uses_existing_data(db_conn):
    client = FakeClient([_response("The median interest rate awarded was 9.0% p.a.")])
    budget = NarrationBudgetTracker(db_conn, cap_usd=5.0)

    result = answer_followup(client, budget, "what is the median interest rate", _answer())

    assert "9.0%" in result
    assert client.calls == 1


def test_answer_followup_raises_on_refusal(db_conn):
    client = FakeClient([_response(None, stop_reason="refusal")])
    budget = NarrationBudgetTracker(db_conn, cap_usd=5.0)

    with pytest.raises(NarrationFailed):
        answer_followup(client, budget, "what is the median interest rate", _answer())


def test_answer_followup_raises_on_api_error(db_conn):
    api_error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    client = FakeClient([api_error])
    budget = NarrationBudgetTracker(db_conn, cap_usd=5.0)

    with pytest.raises(NarrationFailed):
        answer_followup(client, budget, "what is the median interest rate", _answer())
