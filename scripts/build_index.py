#!/usr/bin/env python3
"""Build the Stage 3 retrieval index for one category: chunk generation
(idempotent) + local embedding + hybrid (BM25 + FAISS) index construction.

No API cost — everything here runs locally. Safe to re-run any time (chunk
generation skips already-chunked docs; the index itself is always rebuilt
fresh, which is fast and free at this corpus size).

Example:
    python scripts/build_index.py --category builder_delay
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from rich.console import Console

from ccpf.config import DEFAULT_DB_PATH, REPO_ROOT
from ccpf.db import connection
from ccpf.index.build import build_index
from ccpf.index.embedder import LocalEmbedder

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "index"


@app.command()
def main(category: str = typer.Option(..., help="Category key, e.g. builder_delay")):
    console.print(f"[cyan]Loading local embedding model...[/cyan]")
    t0 = time.time()
    embedder = LocalEmbedder()
    console.print(f"Model loaded in {time.time()-t0:.1f}s (dim={embedder.dim})")

    with connection(DEFAULT_DB_PATH) as conn:
        t0 = time.time()
        summary = build_index(conn, category, embedder, DEFAULT_INDEX_DIR)
        elapsed = time.time() - t0

    console.print(f"[bold]Category:[/bold] {summary.category}")
    console.print(f"[bold]Docs newly chunked:[/bold] {summary.docs_chunked}")
    console.print(f"[bold]Chunks newly created:[/bold] {summary.chunks_created}")
    console.print(f"[bold]Total chunks indexed:[/bold] {summary.chunks_total}")
    console.print(f"[bold]Build time:[/bold] {elapsed:.1f}s")
    console.print(f"[bold]Index saved to:[/bold] {DEFAULT_INDEX_DIR / category}")


if __name__ == "__main__":
    app()
