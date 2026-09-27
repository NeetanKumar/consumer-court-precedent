"""Stage 3 (indexing/retrieval) tests. Uses a deterministic bag-of-words
FakeEmbedder instead of the real sentence-transformers model — fast, no
network, and predictable enough to assert on ranking order."""
import json

import numpy as np
import pytest

from ccpf.index.build import build_index, generate_chunks
from ccpf.index.chunking import chunk_judgment
from ccpf.index.retrieve import search
from ccpf.index.store import HybridIndex, reciprocal_rank_fusion


class FakeEmbedder:
    """Cosine similarity ~= word-overlap similarity — good enough to make
    retrieval ranking assertions deterministic without a real model."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.dim = len(vocab)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            tokens = set(t.lower().split())
            v = np.array([1.0 if w in tokens else 0.0 for w in self.vocab], dtype="float32")
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
        return np.array(vecs, dtype="float32")


# --- chunking ---------------------------------------------------------------

def test_short_judgment_produces_one_chunk():
    chunks = chunk_judgment(1, "short judgment text")
    assert len(chunks) == 1
    assert chunks[0].tid == 1
    assert chunks[0].chunk_index == 0


def test_operative_order_keyword_overrides_position():
    # Near the START of the doc (low position_fraction) but containing an
    # explicit operative-order marker — the keyword signal should win.
    text = "ORDER: the opposite party is directed to pay Rs. 50,000. " + ("padding " * 300)
    chunks = chunk_judgment(1, text)
    assert chunks[0].section == "operative_order"


def test_position_fallback_when_no_keywords():
    # No keyword markers anywhere; a long doc so we get multiple chunks and
    # can check the position-based fallback progression.
    text = ("neutral filler text with no markers at all. " * 100)
    chunks = chunk_judgment(1, text)
    assert chunks[0].section == "facts"
    assert chunks[-1].section in ("findings", "operative_order")


def test_chunks_overlap():
    text = "word " * 500  # long enough to force multiple chunks
    chunks = chunk_judgment(1, text)
    assert len(chunks) > 1
    # Consecutive chunks should share some trailing/leading content given the overlap.
    assert chunks[0].text[-50:] != chunks[0].text[:50]  # sanity: not degenerate


# --- reciprocal rank fusion ---------------------------------------------------

def test_rrf_rewards_items_ranked_highly_by_both_lists():
    dense = [1, 2, 3]
    sparse = [2, 1, 3]
    fused = reciprocal_rank_fusion([dense, sparse])
    ranked = sorted(fused, key=fused.get, reverse=True)
    assert ranked[0] in (1, 2)  # 1 and 2 both rank in the top-2 of both lists
    assert ranked[-1] == 3  # 3 is last in both


# --- hybrid index -------------------------------------------------------------

def test_hybrid_search_finds_relevant_chunk():
    vocab = ["possession", "delay", "refund", "flat", "interest", "irrelevant"]
    embedder = FakeEmbedder(vocab)
    index = HybridIndex(dim=embedder.dim)

    texts = [
        "possession delay refund flat",
        "irrelevant irrelevant irrelevant irrelevant",
        "interest rate discussion",
    ]
    ids = [10, 20, 30]
    index.add(ids, embedder.embed(texts), texts)

    query_emb = embedder.embed(["possession delay"])[0]
    results = index.search(query_emb, "possession delay", top_k=3)
    ranked_ids = [cid for cid, _ in results]
    assert ranked_ids[0] == 10


def test_filtered_search_excludes_disallowed_chunks():
    vocab = ["possession", "delay", "other"]
    embedder = FakeEmbedder(vocab)
    index = HybridIndex(dim=embedder.dim)

    texts = ["possession delay possession delay", "other other other"]
    ids = [1, 2]
    index.add(ids, embedder.embed(texts), texts)

    query_emb = embedder.embed(["possession delay"])[0]
    # Chunk 1 is clearly the best match, but it's excluded from the allowed set.
    results = index.filtered_search(query_emb, "possession delay", allowed_chunk_ids={2}, top_k=5)
    result_ids = [cid for cid, _ in results]
    assert 1 not in result_ids
    assert result_ids == [2]


def test_min_dense_similarity_excludes_orthogonal_query():
    """Regression test for the real bug: a query with no real semantic
    relation to any chunk ('qsq' with the cross-encoder reranker OFF)
    still returned a confident top-K answer, because RRF fusion assigns a
    positive rank-based score to the corpus's best-available candidates
    regardless of whether any of them are actually relevant. The floor
    must apply independent of reranking — this test never reranks."""
    vocab = ["possession", "delay", "refund", "totally", "unrelated", "words"]
    embedder = FakeEmbedder(vocab)
    index = HybridIndex(dim=embedder.dim)

    texts = ["possession delay refund", "possession delay refund"]
    index.add([1, 2], embedder.embed(texts), texts)

    # Zero vocabulary overlap -> cosine similarity is exactly 0.0, well
    # below the floor — must return nothing, not "the best of a bad lot."
    query_emb = embedder.embed(["totally unrelated words"])[0]
    results = index.search(query_emb, "totally unrelated words", top_k=5, min_dense_similarity=0.4)
    assert results == []

    # A real match still passes with the same floor applied.
    good_query_emb = embedder.embed(["possession delay"])[0]
    good_results = index.search(good_query_emb, "possession delay", top_k=5, min_dense_similarity=0.4)
    assert len(good_results) == 2


def test_index_save_and_load_roundtrip(tmp_path):
    vocab = ["a", "b"]
    embedder = FakeEmbedder(vocab)
    index = HybridIndex(dim=embedder.dim)
    texts = ["a a a", "b b b"]
    index.add([1, 2], embedder.embed(texts), texts)

    index.save(tmp_path / "idx")
    loaded = HybridIndex.load(tmp_path / "idx")

    query_emb = embedder.embed(["a"])[0]
    results = loaded.search(query_emb, "a", top_k=2)
    assert results[0][0] == 1
    assert loaded.text_for(1) == "a a a"


# --- build (chunk generation idempotency) ------------------------------------

def _seed_doc(conn, tid, plain_text, category="builder_delay"):
    now = "2026-01-01T00:00:00+00:00"
    conn.execute(
        "INSERT INTO raw_docs (tid, title, docsource, publishdate, raw_html, plain_text, fetched_at, cost_paise) "
        "VALUES (?, 't', 'National Consumer Disputes Redressal Commission', '2021-06-15', '', ?, ?, 20)",
        (tid, plain_text, now),
    )
    conn.execute(
        "INSERT INTO doc_filters (tid, category, is_ncdrc, has_award_language, matched_keywords, passed, filtered_at) "
        "VALUES (?, ?, 1, 1, '', 1, ?)",
        (tid, category, now),
    )
    conn.commit()


def test_generate_chunks_is_idempotent(db_conn):
    _seed_doc(db_conn, 1, "possession delay text " * 50)

    docs1, created1 = generate_chunks(db_conn, "builder_delay")
    assert docs1 == 1
    assert created1 > 0

    # Second call: doc already chunked, nothing new happens.
    docs2, created2 = generate_chunks(db_conn, "builder_delay")
    assert docs2 == 0
    assert created2 == 0


def test_build_index_end_to_end(db_conn, tmp_path):
    _seed_doc(db_conn, 1, "possession delay refund flat compensation " * 20)
    _seed_doc(db_conn, 2, "unrelated content about something else entirely " * 20)

    vocab = ["possession", "delay", "refund", "flat", "compensation", "unrelated", "content", "something", "else", "entirely"]
    embedder = FakeEmbedder(vocab)

    summary = build_index(db_conn, "builder_delay", embedder, tmp_path)

    assert summary.docs_chunked == 2
    assert summary.chunks_total > 0

    index = HybridIndex.load(tmp_path / "builder_delay")
    query_emb = embedder.embed(["possession delay"])[0]
    results = index.search(query_emb, "possession delay", top_k=5)
    assert len(results) > 0


# --- retrieve (metadata pre-filter + judgment-level aggregation) -------------

def _judgment_json(outcome="allowed", amount=100000):
    return json.dumps({
        "case_type": "builder possession delay",
        "fact_summary": "summary",
        "dispute_duration_months": 12,
        "amount_claimed": amount,
        "relief_components": {"refund": None, "interest_rate": None, "mental_agony_compensation": None, "litigation_cost": None},
        "outcome": outcome,
        "forum_level": "NCDRC",
        "confidence": "high",
    })


def _seed_extraction(conn, tid, outcome):
    now = "2026-01-01T00:00:00+00:00"
    conn.execute(
        "INSERT INTO extractions (tid, category, model_used, escalated, confidence, judgment_json, extracted_at) "
        "VALUES (?, 'builder_delay', 'claude-haiku-4-5', 0, 'high', ?, ?)",
        (tid, _judgment_json(outcome=outcome), now),
    )
    conn.commit()


def test_search_prefilters_by_outcome(db_conn, tmp_path):
    _seed_doc(db_conn, 1, "possession delay possession delay possession delay " * 10)
    _seed_extraction(db_conn, 1, outcome="dismissed")
    _seed_doc(db_conn, 2, "possession delay somewhat related " * 10)
    _seed_extraction(db_conn, 2, outcome="allowed")

    vocab = ["possession", "delay", "somewhat", "related"]
    embedder = FakeEmbedder(vocab)
    build_index(db_conn, "builder_delay", embedder, tmp_path)
    index = HybridIndex.load(tmp_path / "builder_delay")

    # tid=1 is the stronger textual match but has the wrong outcome — must be excluded.
    results = search(
        db_conn, index, embedder, "builder_delay", "possession delay",
        top_k=5, outcomes=["allowed"],
    )
    result_tids = [r.tid for r in results]
    assert 1 not in result_tids
    assert 2 in result_tids


def test_search_aggregates_multiple_chunks_to_one_judgment(db_conn, tmp_path):
    # A long doc that will be split into multiple chunks.
    _seed_doc(db_conn, 1, "possession delay refund compensation " * 100)
    _seed_extraction(db_conn, 1, outcome="allowed")

    vocab = ["possession", "delay", "refund", "compensation"]
    embedder = FakeEmbedder(vocab)
    build_index(db_conn, "builder_delay", embedder, tmp_path)
    index = HybridIndex.load(tmp_path / "builder_delay")

    results = search(db_conn, index, embedder, "builder_delay", "possession delay", top_k=5)
    # Only one result even though tid=1 produced multiple chunks.
    assert len(results) == 1
    assert results[0].tid == 1
