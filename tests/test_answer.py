"""Stage 4 (answer generation) tests. Aggregation is pure-function tested
directly; the confidence gate and end-to-end flow reuse the FakeEmbedder
pattern from test_index.py (no real model, no API calls)."""
import json

from ccpf.answer.aggregate import compute_component_stats, compute_outcome_distribution
from ccpf.answer.generate import generate_answer
from ccpf.index.build import build_index
from ccpf.index.store import HybridIndex
from tests.test_index import FakeEmbedder, _seed_doc, _seed_extraction


class FakeReranker:
    """Assigns a fixed score to every candidate — lets tests simulate
    'retrieval found candidates but none of them are actually relevant'
    without downloading the real cross-encoder model."""

    def __init__(self, score: float):
        self._score = score

    def rerank(self, query, candidates, top_k=10):
        return [(cid, self._score) for cid, _ in candidates][:top_k]


# --- aggregate.py (pure functions) -------------------------------------------

def test_component_stats_computes_median_min_max_over_non_null_only():
    judgments = [
        {"tid": 1, "judgment": {"amount_claimed": 100}},
        {"tid": 2, "judgment": {"amount_claimed": 200}},
        {"tid": 3, "judgment": {"amount_claimed": None}},
        {"tid": 4, "judgment": {"amount_claimed": 300}},
    ]
    stats = compute_component_stats("amount_claimed", judgments, lambda j: j["amount_claimed"], sample_size=4)

    assert stats.sample_size == 4
    assert stats.coverage == 3  # excludes the null
    assert stats.median == 200
    assert stats.min == 100
    assert stats.max == 300
    assert stats.contributing_tids == [1, 2, 4]  # excludes tid=3, the null one


def test_component_stats_all_null_reports_no_coverage():
    judgments = [{"tid": 1, "judgment": {"x": None}}, {"tid": 2, "judgment": {"x": None}}]
    stats = compute_component_stats("x", judgments, lambda j: j["x"], sample_size=2)

    assert stats.coverage == 0
    assert stats.median is None
    assert stats.contributing_tids == []
    assert stats.coverage_fraction == 0.0


def test_outcome_distribution_counts_each_outcome():
    judgments = [
        {"tid": 1, "judgment": {"outcome": "allowed"}},
        {"tid": 2, "judgment": {"outcome": "allowed"}},
        {"tid": 3, "judgment": {"outcome": "dismissed"}},
    ]
    dist = compute_outcome_distribution(judgments)
    assert dist == {"allowed": 2, "dismissed": 1}


# --- generate_answer: confidence gate + end-to-end --------------------------

def _seed_corpus(conn, n_docs, outcome="allowed"):
    for tid in range(1, n_docs + 1):
        _seed_doc(conn, tid, f"possession delay refund compensation doc {tid} " * 20)
        _seed_extraction(conn, tid, outcome=outcome)


def test_refuses_below_min_sample_size(db_conn, tmp_path):
    _seed_corpus(db_conn, n_docs=5)  # below default min_sample_size=10

    vocab = ["possession", "delay", "refund", "compensation"]
    embedder = FakeEmbedder(vocab)
    build_index(db_conn, "builder_delay", embedder, tmp_path)
    index = HybridIndex.load(tmp_path / "builder_delay")

    answer = generate_answer(db_conn, index, embedder, "builder_delay", "possession delay")

    assert answer.refused is True
    assert answer.sample_size == 5
    assert "insufficient precedent" in answer.refusal_reason.lower()
    assert answer.amount_claimed_stats is None  # no stats computed on a refusal


def test_answers_at_or_above_min_sample_size(db_conn, tmp_path):
    _seed_corpus(db_conn, n_docs=12)

    vocab = ["possession", "delay", "refund", "compensation"]
    embedder = FakeEmbedder(vocab)
    build_index(db_conn, "builder_delay", embedder, tmp_path)
    index = HybridIndex.load(tmp_path / "builder_delay")

    answer = generate_answer(db_conn, index, embedder, "builder_delay", "possession delay", sample_pool_size=30)

    assert answer.refused is False
    assert answer.sample_size == 12
    assert answer.amount_claimed_stats is not None
    assert answer.amount_claimed_stats.sample_size == 12
    assert len(answer.citations) == 5  # default num_citations
    assert all(c.tid for c in answer.citations)


def test_every_claim_is_traceable_to_contributing_tids(db_conn, tmp_path):
    _seed_corpus(db_conn, n_docs=10)

    vocab = ["possession", "delay", "refund", "compensation"]
    embedder = FakeEmbedder(vocab)
    build_index(db_conn, "builder_delay", embedder, tmp_path)
    index = HybridIndex.load(tmp_path / "builder_delay")

    answer = generate_answer(db_conn, index, embedder, "builder_delay", "possession delay")

    assert answer.refused is False
    # amount_claimed=100000 for every seeded doc (test's _judgment_json default) —
    # so every retrieved judgment should contribute to the stat.
    assert len(answer.amount_claimed_stats.contributing_tids) == answer.amount_claimed_stats.coverage
    assert answer.amount_claimed_stats.coverage == answer.sample_size
    assert set(answer.amount_claimed_stats.contributing_tids).issubset(range(1, 11))


# --- relevance floor: regression test for the "23 lac" bug -------------------
# A real run against the live app showed a low-information query ("23 lac")
# still clearing the sample-size gate and producing a confident-looking
# answer, because retrieval always returns its top-K best-AVAILABLE
# candidates even when none of them are good matches. The fix filters out
# low cross-encoder-score candidates before the sample-size check runs.

def test_low_relevance_candidates_are_filtered_before_sample_size_check(db_conn, tmp_path):
    _seed_corpus(db_conn, n_docs=20)  # plenty by count, but...

    vocab = ["possession", "delay", "refund", "compensation"]
    embedder = FakeEmbedder(vocab)
    build_index(db_conn, "builder_delay", embedder, tmp_path)
    index = HybridIndex.load(tmp_path / "builder_delay")

    # ...every candidate scores below the relevance floor once reranked.
    low_score_reranker = FakeReranker(score=-5.0)
    answer = generate_answer(
        db_conn, index, embedder, "builder_delay", "possession delay",
        reranker=low_score_reranker, min_relevance_score=0.0,
    )

    assert answer.refused is True
    assert "relevant" in answer.refusal_reason.lower()
    assert answer.sample_size == 0


def test_high_relevance_candidates_pass_the_floor(db_conn, tmp_path):
    _seed_corpus(db_conn, n_docs=12)

    vocab = ["possession", "delay", "refund", "compensation"]
    embedder = FakeEmbedder(vocab)
    build_index(db_conn, "builder_delay", embedder, tmp_path)
    index = HybridIndex.load(tmp_path / "builder_delay")

    high_score_reranker = FakeReranker(score=5.0)
    answer = generate_answer(
        db_conn, index, embedder, "builder_delay", "possession delay",
        reranker=high_score_reranker, min_relevance_score=0.0,
    )

    assert answer.refused is False
    assert answer.sample_size == 12


def test_relevance_floor_not_applied_without_a_reranker(db_conn, tmp_path):
    # RRF fusion scores (no-rerank path) aren't on the same scale as
    # cross-encoder scores — the floor must be a no-op here, not silently
    # misapplied to unrelated numbers.
    _seed_corpus(db_conn, n_docs=12)

    vocab = ["possession", "delay", "refund", "compensation"]
    embedder = FakeEmbedder(vocab)
    build_index(db_conn, "builder_delay", embedder, tmp_path)
    index = HybridIndex.load(tmp_path / "builder_delay")

    answer = generate_answer(
        db_conn, index, embedder, "builder_delay", "possession delay",
        reranker=None, min_relevance_score=0.0,
    )

    assert answer.refused is False
    assert answer.sample_size == 12


# --- cross-encoder gate disabled by default: regression test for the -------
# "23 years... 3bhk... broker..." bug. A real, on-topic query got falsely
# refused because the cross-encoder scored every genuinely relevant
# candidate negative — its raw score scale wasn't reliably calibrated
# across phrasing diversity, even though dense retrieval correctly found
# real precedent judgments. The fix: MIN_RELEVANCE_SCORE defaults to None,
# so a reranker returning negative scores no longer causes a false refusal
# by default — only explicit opt-in (tested above) still applies it.

def test_negative_cross_encoder_scores_do_not_cause_false_refusal_by_default(db_conn, tmp_path):
    _seed_corpus(db_conn, n_docs=12)

    vocab = ["possession", "delay", "refund", "compensation"]
    embedder = FakeEmbedder(vocab)
    build_index(db_conn, "builder_delay", embedder, tmp_path)
    index = HybridIndex.load(tmp_path / "builder_delay")

    # A reranker that scores every candidate negative, like the real
    # ms-marco cross-encoder did for the genuinely relevant judgments in
    # the "23 years... 3bhk... broker..." case.
    all_negative_reranker = FakeReranker(score=-2.0)
    answer = generate_answer(
        db_conn, index, embedder, "builder_delay", "possession delay",
        reranker=all_negative_reranker,  # min_relevance_score not passed — must use the None default
    )

    assert answer.refused is False
    assert answer.sample_size == 12
