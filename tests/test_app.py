"""Smoke tests for app.py via streamlit's AppTest. No network, no models:
the DB is a temp file, narration has no API key, and only render paths that
need no retrieval are exercised."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

from ccpf.answer.schema import Citation, ComponentStats, PrecedentAnswer
from ccpf.config import Settings
from ccpf.db import connection
from ccpf.ui import store

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, db_path):
    import ccpf.config as cfg

    monkeypatch.setattr(cfg, "DEFAULT_DB_PATH", db_path)
    monkeypatch.setattr(cfg, "get_settings", lambda: Settings(anthropic_api_key=None, indiankanoon_api_token=None))


def _answer(refused=False):
    if refused:
        return PrecedentAnswer(query="q", filters_applied={}, sample_size=2, refused=True, refusal_reason="too few")
    refund = ComponentStats(
        field_name="refund", sample_size=10, coverage=8, median=1_000_000, min=1e5, max=5e6,
        p25=5e5, p75=2e6, contributing_tids=list(range(8)), values=[1e5, 3e5, 5e5, 8e5, 1e6, 2e6, 3e6, 5e6],
    )
    return PrecedentAnswer(
        query="q", filters_applied={}, sample_size=10, refused=False,
        outcome_distribution={"allowed": 7, "dismissed": 3}, relief_component_stats={"refund": refund},
        citations=[Citation(tid=1, outcome="allowed", amount_claimed=1e6, relief_components={"refund": 1e6},
                            fact_summary="Builder delayed.", relevance_score=0.7, title="A v. B", forum_level="NCDRC")],
    )


def _run(url_params=None):
    at = AppTest.from_file(APP, default_timeout=30)
    for k, v in (url_params or {}).items():
        at.query_params[k] = v
    return at.run()


def test_empty_state_shows_examples_and_no_errors(db_path):
    with connection(db_path) as conn:
        conn.execute("SELECT 1")
    at = _run()
    assert not at.exception
    assert len(at.button) >= 4  # 3 examples + New chat
    assert any("Builder delayed possession" in b.label for b in at.button)


def test_saved_conversation_replays_with_metrics_and_actions(db_path):
    with connection(db_path) as conn:
        cid = store.create_conversation(conn, "owner1", "my question")
        store.add_message(conn, cid, "user", "my question")
        store.add_message(conn, cid, "assistant", "Here is the comparison.", answer_json=_answer().model_dump_json())
    at = _run({"u": "owner1", "c": cid})
    assert not at.exception
    assert any(m.label == "Comparable judgments" for m in at.metric)
    assert any("Here is the comparison." in m.value for m in at.markdown)
    assert any(b.key and b.key.startswith("regen_") for b in at.button)


def test_refusal_replays_as_warning(db_path):
    with connection(db_path) as conn:
        cid = store.create_conversation(conn, "owner1", "q")
        store.add_message(conn, cid, "user", "q")
        store.add_message(conn, cid, "assistant", "", answer_json=_answer(refused=True).model_dump_json())
    at = _run({"u": "owner1", "c": cid})
    assert not at.exception
    assert len(at.warning) == 1


def test_other_visitors_cannot_open_a_conversation(db_path):
    with connection(db_path) as conn:
        cid = store.create_conversation(conn, "owner1", "secret question")
        store.add_message(conn, cid, "user", "secret question")
    at = _run({"u": "intruder", "c": cid})
    assert not at.exception
    assert not any("secret question" in m.value for m in at.markdown)
