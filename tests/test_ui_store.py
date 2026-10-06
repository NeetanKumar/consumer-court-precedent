from ccpf.answer.schema import ComponentStats, PrecedentAnswer
from ccpf.ui import store
from ccpf.ui.suggestions import follow_up_suggestions


def test_conversation_roundtrip_and_title(db_conn):
    cid = store.create_conversation(db_conn, "alice", "  builder   delayed possession " + "x" * 100)
    assert len(store.list_conversations(db_conn, "alice")[0]["title"]) <= store.TITLE_MAX
    store.add_message(db_conn, cid, "user", "hi")
    mid = store.add_message(db_conn, cid, "assistant", "hello", answer_json="{}")
    msgs = store.load_messages(db_conn, "alice", cid)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["id"] == mid


def test_owner_scoping_blocks_other_visitors(db_conn):
    cid = store.create_conversation(db_conn, "alice", "q")
    mid = store.add_message(db_conn, cid, "assistant", "a")
    assert store.list_conversations(db_conn, "bob") == []
    assert store.load_messages(db_conn, "bob", cid) == []
    store.set_feedback(db_conn, "bob", mid, 1)  # must be a no-op
    assert store.load_messages(db_conn, "alice", cid)[0]["feedback"] is None
    store.delete_conversation(db_conn, "bob", cid)
    assert store.owns(db_conn, "alice", cid)


def test_feedback_and_delete(db_conn):
    cid = store.create_conversation(db_conn, "alice", "q")
    mid = store.add_message(db_conn, cid, "assistant", "a")
    store.set_feedback(db_conn, "alice", mid, 0)
    assert store.load_messages(db_conn, "alice", cid)[0]["feedback"] == 0
    store.delete_message(db_conn, "alice", cid, mid)
    assert store.load_messages(db_conn, "alice", cid) == []
    store.delete_conversation(db_conn, "alice", cid)
    assert store.list_conversations(db_conn, "alice") == []


def _stats(name, median, coverage, n=10):
    return ComponentStats(field_name=name, sample_size=n, coverage=coverage, median=median)


def test_suggestions_only_offer_what_the_data_supports():
    answer = PrecedentAnswer(
        query="q", filters_applied={}, sample_size=10, refused=False,
        outcome_distribution={"allowed": 8, "dismissed": 2},
        relief_component_stats={
            "interest_rate": _stats("interest_rate", 9.0, 8),
            "refund": _stats("refund", 1e6, 3),
            "mental_agony_compensation": _stats("mental_agony_compensation", None, 0),
        },
    )
    chips = follow_up_suggestions(answer)
    assert "What interest rate was usually awarded?" in chips
    assert not any("mental agony" in c for c in chips)
    assert len(chips) <= 3


def test_refusal_suggests_example_situations():
    answer = PrecedentAnswer(query="q", filters_applied={}, sample_size=0, refused=True, refusal_reason="x")
    assert all("builder" in c.lower() or "developer" in c.lower() for c in follow_up_suggestions(answer))
