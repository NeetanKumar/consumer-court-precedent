"""All HTTP is mocked via respx — these tests never touch the network or
spend real credit, matching the verified API contract (POST, 0-based
pagenum, Authorization: Token header)."""
import httpx
import pytest
import respx

from ccpf.ingest.client import IndianKanoonClient, IndianKanoonError
from ccpf.models import SearchResponse
from tests.conftest import load_fixture


@respx.mock
def test_search_uses_post_and_token_header():
    route = respx.post("https://api.indiankanoon.org/search/").mock(
        return_value=httpx.Response(200, json=load_fixture("search_page0.json"))
    )
    client = IndianKanoonClient(token="fake-token", rate_limit_per_sec=1000)
    resp = client.search("possession delay", pagenum=0)

    assert route.called
    request = route.calls[0].request
    assert request.method == "POST"
    assert request.headers["Authorization"] == "Token fake-token"
    assert resp.found == 3
    assert len(resp.docs) == 3
    assert resp.docs[0].tid == 111111


@respx.mock
def test_doc_fetches_full_document():
    respx.post("https://api.indiankanoon.org/doc/111111/").mock(
        return_value=httpx.Response(200, json=load_fixture("doc_111111.json"))
    )
    client = IndianKanoonClient(token="fake-token", rate_limit_per_sec=1000)
    doc = client.doc(111111)

    assert doc.tid == 111111
    assert "National Consumer" in doc.docsource
    assert "compensation" in doc.doc


@respx.mock
def test_4xx_is_not_retried():
    route = respx.post("https://api.indiankanoon.org/doc/999/").mock(
        return_value=httpx.Response(401, text="invalid token")
    )
    client = IndianKanoonClient(token="bad-token", rate_limit_per_sec=1000, max_retries=5)

    with pytest.raises(IndianKanoonError):
        client.doc(999)

    # Exactly one attempt — retrying a 4xx just wastes time/credit for nothing.
    assert route.call_count == 1


def test_found_parses_live_api_display_string():
    """Regression test: the published API docs say `found` is an integer,
    but the live API actually returns a display string like
    "1 - 10 of 195632" (verified directly against a real call), or the
    literal "No matching results" for zero hits."""
    assert SearchResponse.model_validate({"found": "1 - 10 of 195632", "docs": []}).found == 195632
    assert SearchResponse.model_validate({"found": "1 - 10 of 1,228", "docs": []}).found == 1228
    assert SearchResponse.model_validate({"found": "No matching results", "docs": []}).found == 0
    assert SearchResponse.model_validate({"found": 42, "docs": []}).found == 42


@respx.mock
def test_search_records_spend_before_model_validation():
    """Regression test: spend must be recorded as soon as the HTTP call
    succeeds, not after our own model_validate — Indian Kanoon bills on
    response, regardless of whether our schema can parse it. A future
    response-shape mismatch must not silently under-count real spend."""
    respx.post("https://api.indiankanoon.org/search/").mock(
        return_value=httpx.Response(200, json={"found": "not a shape we understand", "docs": []})
    )
    client = IndianKanoonClient(token="fake-token", rate_limit_per_sec=1000)
    recorded = []
    with pytest.raises(Exception):
        client.search("x", pagenum=0, on_success=lambda data: recorded.append(data))
    assert len(recorded) == 1  # on_success fired even though model_validate then raised


@respx.mock
def test_5xx_is_retried_then_succeeds():
    route = respx.post("https://api.indiankanoon.org/doc/111111/").mock(
        side_effect=[
            httpx.Response(503, text="temporarily unavailable"),
            httpx.Response(200, json=load_fixture("doc_111111.json")),
        ]
    )
    client = IndianKanoonClient(
        token="fake-token", rate_limit_per_sec=1000,
        max_retries=3, retry_min_wait_s=0.01, retry_max_wait_s=0.02,
    )
    doc = client.doc(111111)

    assert route.call_count == 2
    assert doc.tid == 111111
