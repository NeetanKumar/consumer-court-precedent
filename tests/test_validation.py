"""Validation-set sampling, independent Opus review, and scoring — no real
Anthropic calls (fake client), no network."""
import json
from types import SimpleNamespace

import pytest

from ccpf.extract.cost import LLMBudgetTracker
from ccpf.extract.reviewer import REVIEWER_MODEL, IndependentReviewer, ReviewFailed
from ccpf.extract.sampling import stratified_sample
from ccpf.extract.schema import Judgment, ReliefComponents
from ccpf.extract.scoring import score_validation_set
from ccpf.extract.validation_run import build_validation_set


def _judgment(outcome="allowed", amount_claimed=100000.0, refund=100000.0, interest_rate=9.0, duration=12):
    return Judgment(
        case_type="builder possession delay",
        fact_summary="summary",
        dispute_duration_months=duration,
        amount_claimed=amount_claimed,
        relief_components=ReliefComponents(refund=refund, interest_rate=interest_rate),
        outcome=outcome,
        forum_level="NCDRC",
        confidence="high",
    )


def _seed_extraction(conn, tid, confidence, outcome, judgment=None):
    now = "2026-01-01T00:00:00+00:00"
    conn.execute(
        "INSERT INTO raw_docs (tid, title, docsource, publishdate, raw_html, plain_text, fetched_at, cost_paise) "
        "VALUES (?, 't', 'National Consumer Disputes Redressal Commission', '2021-01-01', '', ?, ?, 20)",
        (tid, f"judgment text {tid}", now),
    )
    j = judgment or _judgment(outcome=outcome)
    conn.execute(
        "INSERT INTO extractions (tid, category, model_used, escalated, confidence, judgment_json, extracted_at) "
        "VALUES (?, 'builder_delay', 'claude-haiku-4-5', 0, ?, ?, ?)",
        (tid, confidence, j.model_dump_json(), now),
    )
    conn.commit()


class FakeReviewClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, model, **kwargs):
        self.calls.append(model)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _response(parsed_output, stop_reason="end_turn", input_tokens=1000, output_tokens=200):
    return SimpleNamespace(
        parsed_output=parsed_output,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# --- sampling -----------------------------------------------------------

def test_all_low_confidence_included(db_conn):
    _seed_extraction(db_conn, 1, "low", "allowed")
    for tid in range(2, 30):
        _seed_extraction(db_conn, tid, "high", "partly_allowed")

    sample = stratified_sample(db_conn, "builder_delay", sample_size=10, seed=1)
    assert any(r.tid == 1 for r in sample)


def test_medium_confidence_capped(db_conn):
    for tid in range(1, 21):
        _seed_extraction(db_conn, tid, "medium", "allowed")

    sample = stratified_sample(db_conn, "builder_delay", sample_size=40, medium_cap=5, seed=1)
    medium_in_sample = [r for r in sample if r.confidence == "medium"]
    assert len(medium_in_sample) == 5


def test_sampling_deterministic_with_seed(db_conn):
    for tid in range(1, 51):
        outcome = ["allowed", "dismissed", "partly_allowed"][tid % 3]
        _seed_extraction(db_conn, tid, "high", outcome)

    sample1 = stratified_sample(db_conn, "builder_delay", sample_size=15, seed=7)
    sample2 = stratified_sample(db_conn, "builder_delay", sample_size=15, seed=7)
    assert [r.tid for r in sample1] == [r.tid for r in sample2]


def test_sampling_covers_rare_outcomes(db_conn):
    for tid in range(1, 21):
        _seed_extraction(db_conn, tid, "high", "partly_allowed")
    _seed_extraction(db_conn, 21, "high", "allowed")
    _seed_extraction(db_conn, 22, "high", "dismissed")

    sample = stratified_sample(db_conn, "builder_delay", sample_size=6, seed=1)
    outcomes_present = {r.outcome for r in sample}
    assert "allowed" in outcomes_present
    assert "dismissed" in outcomes_present


# --- reviewer -------------------------------------------------------------

def test_reviewer_records_spend_and_returns_judgment(db_conn):
    client = FakeReviewClient([_response(_judgment())])
    budget = LLMBudgetTracker(db_conn, cap_usd=5.0)
    reviewer = IndependentReviewer(client, budget)

    result = reviewer.review(tid=1, plain_text="text")

    assert result.judgment.outcome == "allowed"
    assert budget.spent_microusd > 0
    assert client.calls == [REVIEWER_MODEL]


def test_reviewer_raises_on_refusal(db_conn):
    client = FakeReviewClient([_response(None, stop_reason="refusal")])
    budget = LLMBudgetTracker(db_conn, cap_usd=5.0)
    reviewer = IndependentReviewer(client, budget)

    with pytest.raises(ReviewFailed):
        reviewer.review(tid=1, plain_text="text")

    # Spend still recorded — the call happened even though it refused.
    assert budget.spent_microusd > 0


# --- build_validation_set idempotency --------------------------------------

def test_build_validation_set_is_idempotent(db_conn):
    _seed_extraction(db_conn, 5, "high", "allowed")
    from ccpf.extract.sampling import SampleRow

    sample = [SampleRow(tid=5, plain_text="text", confidence="high", outcome="allowed")]
    client = FakeReviewClient([_response(_judgment())])
    budget = LLMBudgetTracker(db_conn, cap_usd=5.0)
    reviewer = IndependentReviewer(client, budget)

    summary1 = build_validation_set(reviewer, db_conn, budget, "builder_delay", sample)
    assert summary1.succeeded == 1

    summary2 = build_validation_set(reviewer, db_conn, budget, "builder_delay", sample)
    assert summary2.skipped_existing == 1
    assert summary2.attempted == 0
    assert client.calls == [REVIEWER_MODEL]  # no second call


# --- scoring ----------------------------------------------------------------

def test_scoring_outcome_and_numeric_fields(db_conn):
    now = "2026-01-01T00:00:00+00:00"
    # Doc A: everything matches.
    _seed_extraction(db_conn, 100, "high", "allowed", judgment=_judgment(outcome="allowed", amount_claimed=100000, refund=100000, duration=12))
    gold_a = _judgment(outcome="allowed", amount_claimed=101000, refund=100000, duration=13)  # within tolerance
    db_conn.execute(
        "INSERT INTO validation_labels (tid, category, reviewer_model, gold_judgment_json, reviewed_at) VALUES (100, 'builder_delay', 'claude-opus-5', ?, ?)",
        (gold_a.model_dump_json(), now),
    )

    # Doc B: outcome mismatch, amount way off.
    _seed_extraction(db_conn, 101, "high", "allowed", judgment=_judgment(outcome="allowed", amount_claimed=50000, duration=6))
    gold_b = _judgment(outcome="dismissed", amount_claimed=500000, duration=6)
    db_conn.execute(
        "INSERT INTO validation_labels (tid, category, reviewer_model, gold_judgment_json, reviewed_at) VALUES (101, 'builder_delay', 'claude-opus-5', ?, ?)",
        (gold_b.model_dump_json(), now),
    )
    db_conn.commit()

    report = score_validation_set(db_conn, "builder_delay")

    assert report.n == 2
    assert report.outcome.matched == 1
    assert report.outcome.mismatched == 1
    assert report.numeric["amount_claimed"].matched == 1  # doc A within 10%
    assert report.numeric["amount_claimed"].mismatched == 1  # doc B way off
    assert report.numeric["dispute_duration_months"].matched == 2  # both within ±2 months
