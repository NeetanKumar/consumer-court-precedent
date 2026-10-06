"""Stage 4 narration tests. No real Anthropic calls — a fake client
stands in, mirroring the pattern from test_extract.py."""
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from ccpf.answer.narrate import (
    NARRATION_MODEL,
    NarrationBudgetExceeded,
    NarrationBudgetTracker,
    NarrationFailed,
    build_narration_payload,
    format_inr,
    narrate_answer,
)
from ccpf.answer.schema import ComponentStats, PrecedentAnswer


# --- format_inr: regression test for the real bug --------------------------
# A real narrated response asked the LLM to "restate in Indian formatting"
# and it mis-grouped digits, turning the source value 238000000
# (correctly ₹23,80,00,000) into ₹2,38,00,00,000 — 10x too large — while
# following every other instruction correctly. Formatting is now done in
# Python and the LLM is told to copy the result verbatim; this test locks
# down that the formatter itself is correct for exactly that value plus
# other edge cases.

def test_format_inr_matches_the_exact_value_that_broke():
    assert format_inr(238000000) == "₹23,80,00,000"


@pytest.mark.parametrize("amount,expected", [
    (0, "₹0"),
    (60, "₹60"),
    (100, "₹100"),
    (50000, "₹50,000"),
    (100000, "₹1,00,000"),  # 1 lakh
    (1000000, "₹10,00,000"),  # 10 lakh
    (10000000, "₹1,00,00,000"),  # 1 crore
    (1979685500, "₹1,97,96,85,500"),
    (-5000, "-₹5,000"),
])
def test_format_inr_edge_cases(amount, expected):
    assert format_inr(amount) == expected


def test_format_inr_rounds_floats():
    assert format_inr(238000000.7) == "₹23,80,00,001"


def test_narration_payload_never_carries_raw_numeric_amounts():
    """Every amount reaching the LLM must already be a pre-formatted
    string — never a raw float/int the model could reformat incorrectly."""
    answer = _answer()
    payload = build_narration_payload(answer)

    assert payload["amount_claimed"]["median"] == "₹10,00,000"
    assert isinstance(payload["amount_claimed"]["median"], str)
    assert payload["amount_claimed"]["min"] == "₹5,00,000"
    assert payload["amount_claimed"]["max"] == "₹20,00,000"


def test_narration_payload_includes_per_citation_awarded_relief():
    """A citation must carry what was actually AWARDED in that specific
    judgment, not just what was claimed — this was previously missing
    entirely, so the narrated text could only ever talk about claims."""
    from ccpf.answer.schema import Citation

    answer = _answer()
    answer.citations = [
        Citation(
            tid=53080839, outcome="partly_allowed", amount_claimed=3629600,
            relief_components={"refund": 3629600, "interest_rate": 9.0, "mental_agony_compensation": None, "litigation_cost": 25000},
            fact_summary="Complainant booked a flat...", relevance_score=1.5,
        )
    ]
    payload = build_narration_payload(answer)

    cited = payload["cited_judgments"][0]
    assert cited["relief_awarded_in_this_case"]["refund"] == "₹36,29,600"
    assert cited["relief_awarded_in_this_case"]["interest_rate"] == "9.0%"
    assert cited["relief_awarded_in_this_case"]["litigation_cost"] == "₹25,000"
    assert cited["relief_awarded_in_this_case"]["mental_agony_compensation"] is None


def _answer(refused=False):
    if refused:
        return PrecedentAnswer(
            query="q", filters_applied={}, sample_size=3, refused=True, refusal_reason="too few"
        )
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


def _response(text, stop_reason="end_turn", input_tokens=500, output_tokens=150):
    return SimpleNamespace(
        content=[FakeTextBlock(text)] if text else [],
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeNarrationClient:
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


def test_refusal_skips_the_llm_entirely(db_conn):
    client = FakeNarrationClient([])  # would raise IndexError if called — proves it wasn't
    budget = NarrationBudgetTracker(db_conn, cap_usd=5.0)

    result = narrate_answer(client, budget, _answer(refused=True))

    assert "Insufficient precedent" in result
    assert client.calls == 0
    assert budget.spent_microusd == 0


def test_narrates_a_real_answer(db_conn):
    client = FakeNarrationClient([_response("Based on 12 similar cases, buyers received around ₹10,00,000...")])
    budget = NarrationBudgetTracker(db_conn, cap_usd=5.0)

    result = narrate_answer(client, budget, _answer())

    assert "₹10,00,000" in result
    assert client.calls == 1
    assert budget.spent_microusd > 0


def test_budget_exceeded_raises_before_calling_llm(db_conn):
    client = FakeNarrationClient([])
    budget = NarrationBudgetTracker(db_conn, cap_usd=0.0)  # zero cap

    with pytest.raises(NarrationBudgetExceeded):
        narrate_answer(client, budget, _answer())

    assert client.calls == 0


def test_narration_refusal_raises_narration_failed(db_conn):
    client = FakeNarrationClient([_response(None, stop_reason="refusal")])
    budget = NarrationBudgetTracker(db_conn, cap_usd=5.0)

    with pytest.raises(NarrationFailed):
        narrate_answer(client, budget, _answer())

    # Spend still recorded — the call happened even though it refused.
    assert budget.spent_microusd > 0


def test_api_error_raises_narration_failed(db_conn):
    api_error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    client = FakeNarrationClient([api_error])
    budget = NarrationBudgetTracker(db_conn, cap_usd=5.0)

    with pytest.raises(NarrationFailed):
        narrate_answer(client, budget, _answer())


def test_budget_tracker_uses_separate_ledger_from_stage2(db_conn):
    from ccpf.extract.cost import LLMBudgetTracker

    # Spend against Stage 2's ledger...
    stage2_budget = LLMBudgetTracker(db_conn, cap_usd=10.0)
    stage2_budget.record("claude-haiku-4-5", ref="x", input_tokens=100000, output_tokens=20000)
    assert stage2_budget.spent_microusd > 0

    # ...must not affect the narration ledger's starting point.
    narration_budget = NarrationBudgetTracker(db_conn, cap_usd=5.0)
    assert narration_budget.spent_microusd == 0


# --- streaming ---------------------------------------------------------------

class _FakeStream:
    def __init__(self, chunks, stop_reason="end_turn", fail_after=None):
        self._chunks, self._stop, self._fail_after = chunks, stop_reason, fail_after

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @property
    def text_stream(self):
        for i, c in enumerate(self._chunks):
            if self._fail_after is not None and i == self._fail_after:
                raise anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))
            yield c

    def get_final_message(self):
        return SimpleNamespace(
            stop_reason=self._stop, usage=SimpleNamespace(input_tokens=1000, output_tokens=100)
        )


class _FakeStreamClient:
    def __init__(self, stream):
        self.messages = SimpleNamespace(stream=lambda **kw: stream)


def test_stream_text_yields_chunks_and_records_spend_once(db_conn):
    from ccpf.answer.narrate import stream_text

    budget = NarrationBudgetTracker(db_conn, cap_usd=1.0)
    client = _FakeStreamClient(_FakeStream(["Hel", "lo"]))
    out = list(stream_text(client, budget, model=NARRATION_MODEL, system="s", user_content="u", max_tokens=10, ref="r"))
    assert "".join(out) == "Hello"
    assert budget.spent_microusd > 0
    n = db_conn.execute("SELECT COUNT(*) FROM narration_spend_log").fetchone()[0]
    assert n == 1


def test_stream_text_refusal_and_empty_raise_narration_failed(db_conn):
    from ccpf.answer.narrate import stream_text

    budget = NarrationBudgetTracker(db_conn, cap_usd=1.0)
    with pytest.raises(NarrationFailed):
        list(stream_text(_FakeStreamClient(_FakeStream(["x"], stop_reason="refusal")), budget,
                         model=NARRATION_MODEL, system="s", user_content="u", max_tokens=10, ref="r"))
    with pytest.raises(NarrationFailed):
        list(stream_text(_FakeStreamClient(_FakeStream([])), budget,
                         model=NARRATION_MODEL, system="s", user_content="u", max_tokens=10, ref="r"))


def test_stream_text_api_error_mid_stream_becomes_narration_failed(db_conn):
    from ccpf.answer.narrate import stream_text

    budget = NarrationBudgetTracker(db_conn, cap_usd=1.0)
    with pytest.raises(NarrationFailed):
        list(stream_text(_FakeStreamClient(_FakeStream(["a", "b"], fail_after=1)), budget,
                         model=NARRATION_MODEL, system="s", user_content="u", max_tokens=10, ref="r"))


def test_stream_text_respects_budget_cap(db_conn):
    from ccpf.answer.narrate import stream_text

    budget = NarrationBudgetTracker(db_conn, cap_usd=0.0)
    budget.record(NARRATION_MODEL, ref="x", input_tokens=1000, output_tokens=1000)
    with pytest.raises(NarrationBudgetExceeded):
        list(stream_text(_FakeStreamClient(_FakeStream(["a"])), budget,
                         model=NARRATION_MODEL, system="s", user_content="u", max_tokens=10, ref="r"))
