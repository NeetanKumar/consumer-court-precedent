"""End-to-end orchestrator test: the resumability property that protects
real credit. All HTTP is mocked via respx (zero spend in test), but the
assertion that matters is behavioral: a second run against the same DB
makes zero additional HTTP calls and records zero additional spend.
"""
import httpx
import respx

from ccpf.config import CategoryConfig, FilterConfig, QueryConfig
from ccpf.ingest.budget import BudgetTracker
from ccpf.ingest.cache import RawCache
from ccpf.ingest.client import IndianKanoonClient
from ccpf.ingest.run import run_ingestion
from tests.conftest import load_fixture

PRICING = {"search": 50, "doc": 20, "docfragment": 5, "docmeta": 2}


def _make_category() -> CategoryConfig:
    return CategoryConfig(
        label="Builder Delay Test",
        budget_inr=100,
        queries=[QueryConfig(id="q1", text="possession delay")],
        filters=FilterConfig(
            ncdrc_source_pattern="national consumer disputes redressal",
            award_amount_patterns=[r"rs\.?\s?[\d,]+"],
            award_verb_patterns=["directed to pay", "compensation"],
        ),
    )


@respx.mock
def test_ingestion_is_resumable_at_zero_cost(db_conn, tmp_path):
    search_route = respx.post("https://api.indiankanoon.org/search/").mock(
        return_value=httpx.Response(200, json=load_fixture("search_page0.json"))
    )
    doc_111111 = respx.post("https://api.indiankanoon.org/doc/111111/").mock(
        return_value=httpx.Response(200, json=load_fixture("doc_111111.json"))
    )
    doc_333333 = respx.post("https://api.indiankanoon.org/doc/333333/").mock(
        return_value=httpx.Response(200, json=load_fixture("doc_333333_no_award.json"))
    )
    # tid 222222 is in the fixture's search results but has no doc mock — a
    # second doc route catches it as a 404 to keep the fixture self-contained.
    respx.post("https://api.indiankanoon.org/doc/222222/").mock(
        return_value=httpx.Response(200, json={
            "tid": 222222, "title": "x",
            "docsource": "National Consumer Disputes Redressal Commission",
            "publishdate": "2019-07-04",
            "doc": "<p>Directed to pay Rs. 5,00,000 compensation to the complainant.</p>",
        })
    )

    category = _make_category()
    cache = RawCache(tmp_path / "raw")
    client = IndianKanoonClient(token="fake-token", rate_limit_per_sec=1000)
    budget = BudgetTracker(db_conn, cap_inr=100, pricing_paise=PRICING)

    summary1 = run_ingestion(client, db_conn, cache, "builder_delay", category, budget)

    assert search_route.call_count == 1
    assert summary1.search_pages_fetched == 1
    assert summary1.docs_fetched == 3  # all three hits from the single search page
    assert summary1.both_passed == 2  # tid 111111 and 222222 pass; 333333 (state, no award) fails
    assert summary1.spend_this_run_inr > 0
    spend_after_first_run = budget.spent_paise

    # --- Simulate a fresh process: new client/budget instances, same DB + cache. ---
    client2 = IndianKanoonClient(token="fake-token", rate_limit_per_sec=1000)
    budget2 = BudgetTracker(db_conn, cap_inr=100, pricing_paise=PRICING)
    summary2 = run_ingestion(client2, db_conn, cache, "builder_delay", category, budget2)

    # The core resumability guarantee: no new HTTP calls, no new spend.
    assert search_route.call_count == 1
    assert doc_111111.call_count == 1
    assert summary2.search_pages_fetched == 0
    assert summary2.docs_fetched == 0
    assert summary2.docs_skipped_existing == 3
    assert summary2.spend_this_run_inr == 0
    assert budget2.spent_paise == spend_after_first_run
