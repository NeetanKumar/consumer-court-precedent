"""Stage 5 (eval/observability) tests: self-retrieval metrics, structural
faithfulness checking, caching, and tracing. Reuses the FakeEmbedder /
seeding helpers from test_index.py — no real model, no API calls."""
import json
import time

from ccpf.answer.generate import generate_answer
from ccpf.eval.cache import get_cached, store_cached
from ccpf.eval.faithfulness import verify_answer
from ccpf.eval.retrieval_metrics import evaluate_self_retrieval
from ccpf.eval.tracing import observed_generate_answer
from ccpf.index.build import build_index
from ccpf.index.store import HybridIndex
from tests.test_index import FakeEmbedder, _seed_doc, _seed_extraction


def _seed_corpus_with_gold(conn, n_docs=12):
    """Each doc gets distinctive vocabulary so self-retrieval has a real
    chance of succeeding or failing meaningfully, not by coincidence."""
    now = "2026-01-01T00:00:00+00:00"
    for tid in range(1, n_docs + 1):
        distinctive_word = f"uniqueterm{tid}"
        text = f"possession delay refund compensation {distinctive_word} " * 20
        _seed_doc(conn, tid, text)
        _seed_extraction(conn, tid, outcome="allowed")
        gold = {
            "case_type": "builder possession delay",
            "fact_summary": f"A complainant in case {distinctive_word} sought possession delay refund",
            "dispute_duration_months": 12,
            "amount_claimed": 100000,
            "relief_components": {"refund": None, "interest_rate": None, "mental_agony_compensation": None, "litigation_cost": None},
            "outcome": "allowed",
            "forum_level": "NCDRC",
            "confidence": "high",
        }
        conn.execute(
            "INSERT INTO validation_labels (tid, category, reviewer_model, gold_judgment_json, reviewed_at) "
            "VALUES (?, 'builder_delay', 'claude-fable-5', ?, ?)",
            (tid, json.dumps(gold), now),
        )
    conn.commit()


def _vocab(n_docs):
    base = ["possession", "delay", "refund", "compensation", "complainant", "sought", "case", "a", "in"]
    return base + [f"uniqueterm{i}" for i in range(1, n_docs + 1)]


# --- retrieval_metrics --------------------------------------------------------

def test_self_retrieval_finds_distinctive_docs(db_conn, tmp_path):
    _seed_corpus_with_gold(db_conn, n_docs=12)
    embedder = FakeEmbedder(_vocab(12))
    build_index(db_conn, "builder_delay", embedder, tmp_path)
    index = HybridIndex.load(tmp_path / "builder_delay")

    report = evaluate_self_retrieval(db_conn, index, embedder, "builder_delay", ks=(1, 5))

    assert report.n_queries == 12
    # Each doc has a unique distinctive term shared only with its own gold
    # summary — retrieval should find it at rank 1 essentially every time.
    assert report.recall_at_k[1] >= 0.9
    assert report.mrr >= 0.9


# --- faithfulness --------------------------------------------------------------

def test_faithfulness_passes_on_real_generated_answer(db_conn, tmp_path):
    _seed_corpus_with_gold(db_conn, n_docs=12)
    embedder = FakeEmbedder(_vocab(12))
    build_index(db_conn, "builder_delay", embedder, tmp_path)
    index = HybridIndex.load(tmp_path / "builder_delay")

    answer = generate_answer(db_conn, index, embedder, "builder_delay", "possession delay")
    report = verify_answer(db_conn, answer, "builder_delay")

    assert report.passed
    assert report.checked_stats > 0


def test_faithfulness_catches_corrupted_median(db_conn, tmp_path):
    _seed_corpus_with_gold(db_conn, n_docs=12)
    embedder = FakeEmbedder(_vocab(12))
    build_index(db_conn, "builder_delay", embedder, tmp_path)
    index = HybridIndex.load(tmp_path / "builder_delay")

    answer = generate_answer(db_conn, index, embedder, "builder_delay", "possession delay")
    corrupted = answer.model_copy(deep=True)
    corrupted.amount_claimed_stats.median = (corrupted.amount_claimed_stats.median or 0) + 999999

    report = verify_answer(db_conn, corrupted, "builder_delay")

    assert not report.passed
    assert any("median" in issue for issue in report.issues)


def test_faithfulness_catches_wrong_citation_outcome(db_conn, tmp_path):
    _seed_corpus_with_gold(db_conn, n_docs=12)
    embedder = FakeEmbedder(_vocab(12))
    build_index(db_conn, "builder_delay", embedder, tmp_path)
    index = HybridIndex.load(tmp_path / "builder_delay")

    answer = generate_answer(db_conn, index, embedder, "builder_delay", "possession delay")
    corrupted = answer.model_copy(deep=True)
    corrupted.citations[0].outcome = "dismissed"  # every seeded doc is actually "allowed"

    report = verify_answer(db_conn, corrupted, "builder_delay")

    assert not report.passed
    assert any("outcome" in issue for issue in report.issues)


def test_faithfulness_skips_refusals():
    from ccpf.answer.schema import PrecedentAnswer

    refusal = PrecedentAnswer(
        query="x", filters_applied={}, sample_size=2, refused=True, refusal_reason="too few"
    )
    report = verify_answer(None, refusal, "builder_delay")  # conn unused on a refusal
    assert report.passed
    assert report.checked_stats == 0


# --- cache ----------------------------------------------------------------------

def test_cache_miss_then_hit(db_conn):
    from ccpf.answer.schema import PrecedentAnswer

    filters = {"outcomes": None, "date_from": None, "date_to": None}
    assert get_cached(db_conn, "builder_delay", "q", filters, 10) is None

    answer = PrecedentAnswer(query="q", filters_applied=filters, sample_size=5, refused=True, refusal_reason="x")
    store_cached(db_conn, "builder_delay", "q", filters, 10, answer)

    cached = get_cached(db_conn, "builder_delay", "q", filters, 10)
    assert cached is not None
    assert cached.sample_size == 5


def test_cache_expires_after_ttl(db_conn):
    from ccpf.answer.schema import PrecedentAnswer

    filters = {"outcomes": None, "date_from": None, "date_to": None}
    answer = PrecedentAnswer(query="q", filters_applied=filters, sample_size=5, refused=True, refusal_reason="x")
    store_cached(db_conn, "builder_delay", "q", filters, 10, answer)

    time.sleep(0.01)
    # A TTL smaller than the elapsed time (fractions of a second) — an
    # immediate expiry check.
    assert get_cached(db_conn, "builder_delay", "q", filters, 10, ttl_hours=0) is None


def test_cache_key_ignores_filter_dict_key_order(db_conn):
    from ccpf.answer.schema import PrecedentAnswer

    answer = PrecedentAnswer(query="q", filters_applied={}, sample_size=5, refused=True, refusal_reason="x")
    store_cached(db_conn, "builder_delay", "q", {"a": 1, "b": 2}, 10, answer)
    cached = get_cached(db_conn, "builder_delay", "q", {"b": 2, "a": 1}, 10)  # different key order
    assert cached is not None


# --- tracing ----------------------------------------------------------------------

def test_observed_generate_answer_logs_and_caches(db_conn, tmp_path):
    _seed_corpus_with_gold(db_conn, n_docs=12)
    embedder = FakeEmbedder(_vocab(12))
    build_index(db_conn, "builder_delay", embedder, tmp_path)
    index = HybridIndex.load(tmp_path / "builder_delay")

    answer1 = observed_generate_answer(db_conn, index, embedder, "builder_delay", "possession delay")
    answer2 = observed_generate_answer(db_conn, index, embedder, "builder_delay", "possession delay")

    assert answer1.sample_size == answer2.sample_size

    log_rows = db_conn.execute("SELECT * FROM query_log").fetchall()
    assert len(log_rows) == 2  # both the miss and the hit get logged

    cache_rows = db_conn.execute("SELECT * FROM query_cache").fetchall()
    assert len(cache_rows) == 1  # same request key -> one cache row, updated not duplicated
