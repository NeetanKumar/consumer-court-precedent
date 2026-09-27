#!/usr/bin/env python3
"""Self-retrieval eval: query with each validation-set judgment's own
(independently-written) gold fact summary, check if the system retrieves
that judgment back. See eval/retrieval_metrics.py for what this can and
can't tell you — it's not recall@k against true multi-document relevance
judgments, which we don't have.

Example:
    python scripts/eval_retrieval.py --category builder_delay
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from rich.console import Console

from ccpf.config import DEFAULT_DB_PATH, REPO_ROOT
from ccpf.db import connection
from ccpf.eval.retrieval_metrics import evaluate_self_retrieval
from ccpf.index.embedder import LocalEmbedder
from ccpf.index.store import HybridIndex

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "index"


@app.command()
def main(category: str = typer.Option(..., help="Category key, e.g. builder_delay")):
    index_path = DEFAULT_INDEX_DIR / category
    if not index_path.exists():
        console.print(f"[red]No index found at {index_path} — run build_index.py first.[/red]")
        raise typer.Exit(1)

    console.print("[cyan]Loading embedder and index...[/cyan]")
    embedder = LocalEmbedder()
    index = HybridIndex.load(index_path)

    with connection(DEFAULT_DB_PATH) as conn:
        report = evaluate_self_retrieval(conn, index, embedder, category)

    if report.n_queries == 0:
        console.print("[yellow]No validation labels found — run build_validation_set.py first.[/yellow]")
        raise typer.Exit(1)

    console.print(f"[bold]Self-retrieval eval[/bold] ({report.n_queries} queries, from validation_labels gold summaries)")
    console.print("[dim]Not recall@k against true relevance judgments — see module docstring.[/dim]\n")
    for k, recall in sorted(report.recall_at_k.items()):
        console.print(f"  Recall@{k}: {recall*100:.0f}%")
    console.print(f"  MRR: {report.mrr:.3f}")
    if report.misses:
        console.print(f"\n[yellow]Never retrieved within top-{max(report.recall_at_k)}:[/yellow] {report.misses}")


if __name__ == "__main__":
    app()
