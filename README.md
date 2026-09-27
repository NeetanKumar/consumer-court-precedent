# Consumer Court Precedent Finder

Answers: *"I'm in situation X (e.g. builder delayed possession) — what
compensation have similar consumers actually been awarded?"* — grounded in
real NCDRC judgments, with citations and amounts.

## Why this isn't a plain RAG demo

Naive chunk-and-embed RAG can't answer this question well. "What's the
median award for a 3-year possession delay?" requires **structured facts
per judgment** — award amount, delay duration, relief components, outcome —
that can be aggregated into a range and a sample size. Semantic similarity
over text chunks retrieves passages that *sound* relevant but can't produce
a median. So this is a **structured-extraction + retrieval hybrid**: an LLM
extracts a typed record per judgment into SQL; retrieval narrows which
judgments to aggregate; the answer is computed from the structured facts,
with every number linked back to its source judgment.

## Status

**Stage 1 (ingestion) is built and tested for one category:** builder/
real-estate possession delay. Stages 2–5 are stubbed as module boundaries
(`src/ccpf/extract`, `index`, `answer`, `eval`) with their planned schemas
documented in each `__init__.py` — not yet implemented.

## Pipeline stages

1. **Ingestion** *(built)* — fetch judgments + metadata from the Indian
   Kanoon API into SQLite, with local NCDRC + award-language filtering.
2. **Structured extraction** *(planned)* — LLM extraction into a typed
   schema (case_type, dispute duration, relief components, outcome),
   validated against a hand-labeled sample.
3. **Indexing** *(planned)* — section-aware chunking, metadata pre-filter,
   hybrid BM25 + dense retrieval, cross-encoder rerank.
4. **Answer generation** *(planned)* — precedent comparison (range/median/
   sample size) with per-claim citations; refuses or caveats below a
   minimum sample size instead of guessing.
5. **Eval & observability** *(planned)* — recall@k/MRR, faithfulness
   checks, latency/cost tracing, a thin-category refusal test.

## Setup

Requires Python **3.10+** (the repo standardizes on 3.12; system Python
3.9 will not work — `faiss-cpu`, needed in Stage 3, requires ≥3.10).

```bash
brew install python@3.12          # if not already installed
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env              # fill in INDIANKANOON_API_TOKEN
```

## Running ingestion (Stage 1)

All commands are run from the repo root with the venv active.

```bash
# 1. Zero-spend correctness check — HTTP is mocked, no API key needed.
pytest tests/ -v

# 2. Cost projection BEFORE spending anything real.
#    Costs one search call per query (page 0, to read `found`), then
#    projects the rest without fetching further pages or documents.
python scripts/ingest.py --category builder_delay --dry-run

# 3. Small real run to sanity-check response shape against the live API.
python scripts/ingest.py --category builder_delay --max-pages 1 --max-docs 10

# 4. Resumability check — the property that protects your credit.
#    Ctrl-C mid-run, then re-run the same command: it must report zero
#    new spend and "N already fetched".
python scripts/ingest.py --category builder_delay --max-docs 10

# 5. Full run under the category's configured budget cap (₹400 for
#    builder_delay, see config.yaml; override with --budget-inr).
python scripts/ingest.py --category builder_delay

# 6. Inspect the corpus: counts, filter pass rates, date spread, spend.
python scripts/inspect_db.py
```

Ingestion is resumable by design: every fetch loop checks SQLite for an
existing row *before* making a paid call and skips it if found. A crash,
`Ctrl-C`, or laptop sleep costs nothing to recover from — re-running picks
up exactly where it stopped. All spend is tracked in integer paise via an
append-only ledger (`spend_log`), read back at startup so the budget cap
is enforced correctly across process restarts, not just within one run.

## API contract notes (Indian Kanoon)

Verified directly against `api.indiankanoon.org/documentation/` — a few
details are easy to get wrong:

- All endpoints are **POST**; GET fails.
- Auth: `Authorization: Token <key>` header.
- `pagenum` in `/search/` is **0-based**.
- **There is no `ncdrc` doctype.** `doctypes:consumer` returns NCDRC plus
  State/District commission matter — NCDRC-only filtering is done locally
  on `docsource` in `ingest/filters.py`, after download.
- Billing is **per page/document returned**, not per call attempted;
  `maxpages` batches round-trips but does not reduce cost.

## Repo layout

```
src/ccpf/
  config.py, db.py, models.py   # shared: settings, SQLite schema, pydantic models
  ingest/                       # Stage 1 — built
  extract/  index/  answer/  eval/   # Stages 2-5 — interface stubs only
scripts/
  ingest.py       # CLI: run ingestion for one category
  inspect_db.py   # CLI: corpus stats / Stage-1 exit gate
tests/            # respx-mocked, zero real spend or network access
```

Each stage is a separate package with its own CLI entry point — ingestion,
extraction, retrieval, and eval are meant to run and be tested
independently, never as one monolithic script.
