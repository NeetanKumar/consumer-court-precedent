"""Local post-download filters: NCDRC-only + award-language checks.

These run purely against text already in the DB — no network involved.
Fixture docs mirror what fetch would actually produce: one genuine NCDRC
award, one State-commission doc with no award language (should fail both).
"""
from ccpf.config import FilterConfig
from ccpf.ingest.filters import DocFilter, run_filters
from ccpf.ingest.htmlutil import html_to_text

FILTER_CONFIG = FilterConfig(
    ncdrc_source_pattern="national consumer disputes redressal",
    award_amount_patterns=[r"₹\s?[\d,]+", r"rs\.?\s?[\d,]+", r"\d+(\.\d+)?\s*(lakh|lakhs|crore|crores)"],
    award_verb_patterns=["awarded", "directed to pay", "compensation of", r"interest\s*@", "refund"],
)


def test_ncdrc_award_doc_passes_both_filters():
    doc_filter = DocFilter("builder_delay", FILTER_CONFIG)
    text = html_to_text(
        "<p>Held: opposite party directed to pay Rs. 12,50,000 as compensation "
        "along with interest @ 9% per annum.</p>"
    )
    verdict = doc_filter.evaluate(
        docsource="National Consumer Disputes Redressal Commission",
        plain_text=text,
        tid=111111,
    )
    assert verdict.is_ncdrc is True
    assert verdict.has_award_language is True
    assert verdict.passed is True
    assert verdict.matched_keywords  # non-empty, useful for debugging pass rate


def test_state_commission_doc_fails_ncdrc_check():
    doc_filter = DocFilter("builder_delay", FILTER_CONFIG)
    text = html_to_text("<p>Matter transferred to district forum. No order on merits.</p>")
    verdict = doc_filter.evaluate(
        docsource="State Consumer Disputes Redressal Commission, Delhi",
        plain_text=text,
        tid=333333,
    )
    assert verdict.is_ncdrc is False
    assert verdict.has_award_language is False
    assert verdict.passed is False


def test_ncdrc_doc_without_award_language_fails():
    doc_filter = DocFilter("builder_delay", FILTER_CONFIG)
    text = html_to_text("<p>The complaint is dismissed for lack of evidence.</p>")
    verdict = doc_filter.evaluate(
        docsource="National Consumer Disputes Redressal Commission",
        plain_text=text,
        tid=444444,
    )
    assert verdict.is_ncdrc is True
    assert verdict.has_award_language is False
    assert verdict.passed is False


def test_run_filters_persists_and_is_idempotent(db_conn):
    db_conn.execute(
        """INSERT INTO raw_docs (tid, title, docsource, publishdate, raw_html, plain_text, fetched_at, cost_paise)
           VALUES (111111, 't', 'National Consumer Disputes Redressal Commission', '2021-03-12',
                   '<p>directed to pay Rs. 50,000 compensation</p>',
                   'directed to pay Rs. 50,000 compensation', 'now', 20)"""
    )
    db_conn.commit()

    verdicts = run_filters(db_conn, "builder_delay", FILTER_CONFIG)
    assert len(verdicts) == 1
    assert verdicts[0].passed is True

    # Re-running must not re-evaluate already-filtered docs (zero-cost, idempotent).
    verdicts_again = run_filters(db_conn, "builder_delay", FILTER_CONFIG)
    assert verdicts_again == []

    row = db_conn.execute(
        "SELECT passed FROM doc_filters WHERE tid = 111111 AND category = 'builder_delay'"
    ).fetchone()
    assert row["passed"] == 1
