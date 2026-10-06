"""Comparability facets: narrow the retrieved set to judgments whose delay
length is close to the user's, instead of blending a 1-year and a 9-year
delay into one median.

Parsing is deliberately plain regex, not an LLM call: it only has to catch
explicit durations ("3 years", "18 months"). If nothing is found the
search is simply unfaceted.
"""
from __future__ import annotations

import re
from typing import Optional

_DURATION = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*[- ]?\s*(years?|yrs?|months?|mos?)\b", re.IGNORECASE
)
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10}
_WORD_DURATION = re.compile(
    r"\b(" + "|".join(_WORD_NUMBERS) + r")\s*[- ]?\s*(years?|yrs?|months?)\b", re.IGNORECASE
)


def parse_duration_months(text: str) -> Optional[int]:
    """First explicit duration in `text`, in months; None if there is none."""
    m = _DURATION.search(text)
    if m:
        value, unit = float(m.group(1)), m.group(2).lower()
    else:
        m = _WORD_DURATION.search(text)
        if not m:
            return None
        value, unit = _WORD_NUMBERS[m.group(1).lower()], m.group(2).lower()
    months = value * 12 if unit.startswith("y") else value
    return int(round(months)) if months > 0 else None


def duration_window(months: int, tolerance: float = 0.4, min_slack: int = 6) -> tuple[int, int]:
    """Inclusive (low, high) month range around `months`: ±40%, at least ±6."""
    slack = max(min_slack, int(round(months * tolerance)))
    return max(0, months - slack), months + slack
