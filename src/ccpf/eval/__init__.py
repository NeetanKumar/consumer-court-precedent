"""Stage 5 — eval & observability.

- retrieval_metrics.py: self-retrieval recall@k / MRR against the Stage 2
  validation set's independently-written gold fact summaries. Not true
  multi-document relevance-judged recall (we don't have that) — an honest,
  narrower proxy: can the system re-find a judgment from a paraphrase of it.
- faithfulness.py: structural consistency check for a PrecedentAnswer —
  the equivalent of RAGAS faithfulness for a system with no LLM narrator
  to hallucination-check; verifies every reported number actually matches
  the DB records it claims to derive from.
- tracing.py: latency/outcome logging per query (query_log table). No cost
  field — Stage 3/4 queries are $0, by design (local embeddings, no LLM
  narration).
- cache.py: persistent (SQLite-backed) cache for identical
  (category, query, filters) requests.
"""
