"""Stage 3 — indexing & retrieval.

Section-aware chunking (chunking.py) tags each chunk facts/findings/
operative_order. Retrieval (store.py, retrieve.py) applies metadata
filters (category, date, outcome) as a genuine PRE-filter — the candidate
set is restricted before any scoring, not scored-then-discarded — then
runs hybrid BM25 + dense retrieval (reciprocal rank fusion), then a
cross-encoder rerank (rerank.py) over the narrowed candidate set.

Embedder (embedder.py) is the swappable interface per the locked-in
decision: local sentence-transformers by default (LocalEmbedder), a
hosted implementation stubbed behind the same interface (HostedEmbedder)
for later A/B comparison in Stage 5.

FAISS is the vector store (no Pinecone, per the local-first preference).
"""
