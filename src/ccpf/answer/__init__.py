"""Stage 4 — answer generation.

generate_answer() (this package) retrieves comparable judgments via
Stage 3 (retrieve.search), then computes a precedent comparison directly
from their Stage 2 structured fields — no LLM call, deterministic. Every
statistic in the output (aggregate.py, schema.py's ComponentStats) carries
the tids that contributed to it, so every claim is traceable to a
specific judgment by construction. (narrate.py, used by app.py, is a
separate, optional layer that restates this already-computed data as
prose via an LLM that never computes anything itself — see its docstring.)

Two independent confidence gates, both required to pass before an answer
is generated instead of a refusal:
  1. Relevance floor — index/retrieve.py's MIN_DENSE_SIMILARITY (0.4 raw
     cosine similarity). Catches queries that don't meaningfully match
     anything in the corpus. Without this, retrieval always returns its
     top-K best-AVAILABLE candidates even for gibberish or off-topic
     input, because rank-based fusion has no notion of "none of these are
     actually relevant."
  2. Sample-size floor (MIN_SAMPLE_SIZE, 10) — catches queries that match
     something real, but too little precedent exists to trust a range.

A cross-encoder-score gate (MIN_RELEVANCE_SCORE) was tried as an
additional relevance check on top of gate 1, but is DISABLED by default —
see its docstring in generate.py. It produced a false refusal on a
genuinely on-topic real query because the cross-encoder's score scale
wasn't reliably calibrated across phrasing diversity, unlike the dense
cosine-similarity floor. Two failure modes have now been found and fixed
by testing against real queries, not just synthetic ones: an earlier
version had only gate 2, and a low-information query ("23 lac") or
literal keyboard mash ("qsq") could clear it using only weakly- or
un-related judgments; the cross-encoder gate added to fix that then
over-corrected and rejected a genuinely relevant query.
"""
