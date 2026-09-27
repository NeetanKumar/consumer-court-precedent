"""BudgetTracker: cap enforcement and spend recovery across restarts.

Money is stored in integer paise throughout — these tests assert on paise,
not float rupees, since that's the actual invariant being protected against
drift.
"""
import pytest

from ccpf.ingest.budget import BudgetExceeded, BudgetTracker

PRICING = {"search": 50, "doc": 20, "docfragment": 5, "docmeta": 2}


def test_check_and_reserve_raises_at_cap(db_conn):
    budget = BudgetTracker(db_conn, cap_inr=0.60, pricing_paise=PRICING)
    # 12 doc calls at 20 paise = 240 paise = 2.40... wait, use search (50p) to hit 0.60 (60p) cap fast
    budget.check_and_reserve("search")  # 50p, within 60p cap
    budget.record_spend("search", 50)
    with pytest.raises(BudgetExceeded):
        budget.check_and_reserve("search")  # would be 100p > 60p cap


def test_spend_survives_reconstruction(db_conn):
    budget1 = BudgetTracker(db_conn, cap_inr=10.0, pricing_paise=PRICING)
    budget1.check_and_reserve("doc")
    budget1.record_spend("doc", 20)
    assert budget1.spent_paise == 20

    # Simulate a fresh process: new tracker instance, same DB connection/file.
    budget2 = BudgetTracker(db_conn, cap_inr=10.0, pricing_paise=PRICING)
    assert budget2.spent_paise == 20


def test_failed_call_is_not_charged(db_conn):
    budget = BudgetTracker(db_conn, cap_inr=10.0, pricing_paise=PRICING)
    cost = budget.check_and_reserve("doc")
    assert cost == 20
    # Caller never calls record_spend because the API call failed.
    assert budget.spent_paise == 0


def test_unknown_endpoint_raises(db_conn):
    budget = BudgetTracker(db_conn, cap_inr=10.0, pricing_paise=PRICING)
    with pytest.raises(ValueError):
        budget.check_and_reserve("notanendpoint")


def test_project_search_cost(db_conn):
    budget = BudgetTracker(db_conn, cap_inr=10.0, pricing_paise=PRICING)
    # 23 results at 10/page -> 3 pages -> 3 * 50p = 150p
    assert budget.project_search_cost(found=23, page_size=10) == 150
    assert budget.project_search_cost(found=0, page_size=10) == 0
