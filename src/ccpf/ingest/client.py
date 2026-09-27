"""Indian Kanoon API client.

Verified contract (api.indiankanoon.org/documentation/, checked directly
rather than assumed):
  - All endpoints are POST. GET requests fail.
  - Auth header: `Authorization: Token <key>`.
  - POST /search/?formInput=<query>&pagenum=<n> — pagenum is 0-BASED.
    Response JSON: {found, docs: [{tid, title, headline, docsource,
    docsize}], categories, encodedformInput}.
  - POST /doc/<tid>/ — full document. Response JSON: {tid, title,
    docsource, publishdate, doc (HTML)}.
  - Billing is per page/document RETURNED, not per call attempted — a
    retried 4xx would still be a wasted round-trip even if not separately
    billed, so retries are restricted to 429/5xx/network errors only.

This module makes no budget decisions itself — BudgetTracker.check_and_reserve
must be called by the orchestrator before every request. The client's job is
purely: talk to the API correctly, rate-limit, retry transient failures, and
log actual spend for every call that succeeds.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ccpf.models import DocResponse, SearchResponse

logger = logging.getLogger(__name__)


class IndianKanoonError(Exception):
    """Non-retryable API error (4xx, malformed response)."""


class IndianKanoonClient:
    def __init__(
        self,
        token: str,
        base_url: str = "https://api.indiankanoon.org",
        rate_limit_per_sec: float = 1.0,
        max_retries: int = 5,
        retry_min_wait_s: float = 2.0,
        retry_max_wait_s: float = 30.0,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._min_interval = 1.0 / rate_limit_per_sec if rate_limit_per_sec > 0 else 0.0
        self._last_call_ts = 0.0
        self._max_retries = max_retries
        self._retry_min_wait_s = retry_min_wait_s
        self._retry_max_wait_s = retry_max_wait_s
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Token {token}"},
            transport=transport,
            timeout=30.0,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "IndianKanoonClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call_ts
        wait = self._min_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    def _post(self, path: str, params: dict) -> dict:
        """POST with rate limiting and retry restricted to transient failures.

        Never retries 4xx — a retried auth error or malformed query just
        burns time (and, for endpoints billed per attempt rather than per
        page, credit) for nothing.
        """

        def _is_retryable(exc: BaseException) -> bool:
            if isinstance(exc, httpx.TransportError):
                return True
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                return status == 429 or status >= 500
            return False

        @retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(min=self._retry_min_wait_s, max=self._retry_max_wait_s),
            reraise=True,
        )
        def _do_request() -> dict:
            self._throttle()
            resp = self._client.post(path, params=params)
            self._last_call_ts = time.monotonic()
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    # 4xx: not retryable, surface immediately as our own error type.
                    raise IndianKanoonError(
                        f"{path} -> {exc.response.status_code}: {exc.response.text[:300]}"
                    ) from exc
                raise
            return resp.json()

        return _do_request()

    def search(self, form_input: str, pagenum: int, on_success=None) -> SearchResponse:
        """POST /search/. pagenum is 0-based per the verified API contract.

        `on_success`, if given, is invoked with the raw response dict right
        after the HTTP call succeeds — BEFORE model validation. Indian
        Kanoon bills as soon as it returns a page, regardless of whether
        our own schema can parse it; recording spend here (not after
        model_validate) is what protects the budget ledger from silently
        under-counting real spend if a future response shape mismatch
        raises a ValidationError.
        """
        data = self._post("/search/", {"formInput": form_input, "pagenum": pagenum})
        if on_success is not None:
            on_success(data)
        return SearchResponse.model_validate(data)

    def doc(self, tid: int, on_success=None) -> DocResponse:
        """POST /doc/<tid>/ — full document text + metadata. See `search()`
        docstring re: `on_success` timing relative to billing."""
        data = self._post(f"/doc/{tid}/", {})
        data.setdefault("tid", tid)
        if on_success is not None:
            on_success(data)
        return DocResponse.model_validate(data)
