#!/usr/bin/env python3
"""Query the Stage 3 retrieval index. Run build_index.py first.

Examples:
    python scripts/search.py --category builder_delay --query "builder delayed possession by 3 years"
    python scripts/search.py --category builder_delay --query "possession delay" --outcome allowed --outcome partly_allowed
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from rich.console import Console
from rich.table import Table

from ccpf.config import DEFAULT_DB_PATH, REPO_ROOT
from ccpf.db import connection
from ccpf.index.embedder import LocalEmbedder
from ccpf.index.rerank import CrossEncoderReranker
from ccpf.index.retrieve import search
from ccpf.index.store import HybridIndex

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "index"


@app.command()
def main(
    category: str = typer.Option(..., help="Category key, e.g. builder_delay"),
    query: str = typer.Option(..., help="Free-text query, e.g. a user's described situation."),
    top_k: int = typer.Option(5, help="Number of judgments to return."),
    outcome: list[str] = typer.Option(None, help="Filter to these outcome(s). Repeatable."),
    date_from: Optional[str] = typer.Option(None, help="Filter: publishdate >= this (YYYY-MM-DD)."),
    date_to: Optional[str] = typer.Option(None, help="Filter: publishdate <= this (YYYY-MM-DD)."),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip the cross-encoder rerank step."),
):
    index_path = DEFAULT_INDEX_DIR / category
    if not index_path.exists():
        console.print(f"[red]No index found at {index_path} — run build_index.py first.[/red]")
        raise typer.Exit(1)

    console.print("[cyan]Loading embedder and index...[/cyan]")
    embedder = LocalEmbedder()
    index = HybridIndex.load(index_path)
    reranker = None if no_rerank else CrossEncoderReranker()

    with connection(DEFAULT_DB_PATH) as conn:
        results = search(
            conn, index, embedder, category, query,
            top_k=top_k, outcomes=outcome or None, date_from=date_from, date_to=date_to,
            reranker=reranker,
        )

    if not results:
        console.print("[yellow]No results — check your filters or that the index was built.[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Top {len(results)} results for: {query!r}")
    table.add_column("tid")
    table.add_column("Score", justify="right")
    table.add_column("Outcome")
    table.add_column("Amount claimed", justify="right")
    table.add_column("Refund awarded", justify="right")
    table.add_column("Section")
    table.add_column("Fact summary")

    for r in results:
        j = r.judgment
        relief = j.get("relief_components", {})
        table.add_row(
            str(r.tid),
            f"{r.score:.3f}",
            j.get("outcome", ""),
            str(j.get("amount_claimed") or ""),
            str(relief.get("refund") or ""),
            r.best_chunk_section,
            (j.get("fact_summary") or "")[:100],
        )
    console.print(table)


if __name__ == "__main__":
    app()
