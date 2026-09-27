"""Pydantic models for API responses and internal records.

These mirror the verified Indian Kanoon response shapes exactly (see
ingest/client.py docstring for the source), plus a couple of internal
bookkeeping models used by the orchestrator.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class SearchHit(BaseModel):
    """One entry in a /search/ response's `docs[]` array."""

    tid: int
    title: str
    headline: str = ""
    docsource: str = ""
    docsize: Optional[int] = None


_FOUND_TOTAL_RE = re.compile(r"of\s+([\d,]+)\s*$")


class SearchResponse(BaseModel):
    """Raw shape of a /search/ response.

    `found` is NOT a plain integer despite the published API docs — the
    live API returns a display string like "1 - 10 of 195632", or the
    literal string "No matching results" when there are zero hits. Verified
    directly against a live call; parsed here into the total int.
    """

    found: int = 0
    docs: list[SearchHit] = Field(default_factory=list)
    categories: Optional[list] = None
    encodedformInput: Optional[str] = None

    @field_validator("found", mode="before")
    @classmethod
    def _parse_found(cls, v: Any) -> int:
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            if v.strip().lower() == "no matching results":
                return 0
            m = _FOUND_TOTAL_RE.search(v)
            if m:
                return int(m.group(1).replace(",", ""))
        raise ValueError(f"Unrecognized 'found' format: {v!r}")


class DocResponse(BaseModel):
    """Raw shape of a /doc/<tid>/ response. `doc` is HTML."""

    tid: int
    title: str = ""
    docsource: str = ""
    publishdate: str = ""
    doc: str = ""


class FilterVerdict(BaseModel):
    """Result of the local post-download NCDRC + award-language check."""

    tid: int
    category: str
    is_ncdrc: bool
    has_award_language: bool
    matched_keywords: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.is_ncdrc and self.has_award_language
