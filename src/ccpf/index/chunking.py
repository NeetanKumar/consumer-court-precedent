"""Section-aware chunking of judgment text.

The compensation-relevant language (amounts, relief granted) concentrates
in the operative order near the end of a judgment — retrieving a chunk
tagged "facts" when the query is really about awarded relief is a common
failure mode of naive fixed-window chunking. This tags each chunk with a
heuristic section using two signals combined: keyword markers (checked
first, since they're a direct signal regardless of position) and a
fallback based on position within the document (operative orders and
findings skew toward the back of a judgment; facts/background skew toward
the front).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150

# Checked first — a direct textual signal beats position guessing.
OPERATIVE_MARKERS = [
    r"\border\b",
    r"in view of (the )?(above|foregoing)",
    r"is/?are directed to",
    r"directed to (pay|refund)",
    r"complaint is (allowed|dismissed|partly[- ]allowed)",
    r"appeal is (allowed|dismissed|disposed)",
    r"it is ordered that",
]
FINDINGS_MARKERS = [
    r"we have (heard|considered)",
    r"after (hearing|considering|perusal|perusing)",
    r"in our (view|opinion)",
    r"we find that",
    r"having heard",
    r"perused the (record|material)",
]


@dataclass
class Chunk:
    tid: int
    chunk_index: int
    section: str  # "facts" | "findings" | "operative_order"
    text: str


def _classify_section(text: str, position_fraction: float) -> str:
    lowered = text.lower()
    if any(re.search(p, lowered) for p in OPERATIVE_MARKERS):
        return "operative_order"
    if any(re.search(p, lowered) for p in FINDINGS_MARKERS):
        return "findings"
    if position_fraction > 0.75:
        return "operative_order"
    if position_fraction > 0.4:
        return "findings"
    return "facts"


def chunk_judgment(
    tid: int,
    plain_text: str,
    chunk_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    n = len(plain_text)
    if n == 0:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < n:
        end = min(start + chunk_chars, n)
        text = plain_text[start:end].strip()
        if text:
            section = _classify_section(text, position_fraction=start / n)
            chunks.append(Chunk(tid=tid, chunk_index=idx, section=section, text=text))
            idx += 1
        if end == n:
            break
        start = end - overlap  # overlap so a fact split across the boundary isn't lost entirely
    return chunks
