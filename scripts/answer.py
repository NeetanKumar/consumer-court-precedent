#!/usr/bin/env python3
"""Stage 4 CLI: ask a situation, get a precedent comparison. No API cost —
retrieval is local (Stage 3), and the answer is computed, not LLM-narrated.

Examples:
    python scripts/answer.py --category builder_delay --situation "builder delayed possession by 3 years, refused refund"
    python scripts/answer.py --category builder_delay --situation "..." --outcome dismissed  # likely triggers refusal
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
from ccpf.eval.faithfulness import verify_answer
from ccpf.eval.tracing import observed_generate_answer
from ccpf.index.embedder import LocalEmbedder
from ccpf.index.rerank import CrossEncoderReranker
from ccpf.index.store import HybridIndex

app = typer.Typer(add_completion=False)
console = Console()

DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "index"


@app.command()
def main(
    category: str = typer.Option(..., help="Category key, e.g. builder_delay"),
    situation: str = typer.Option(..., help="Free-text description of the user's situation."),
    outcome: list[str] = typer.Option(None, help="Filter to these outcome(s). Repeatable."),
    date_from: Optional[str] = typer.Option(None, help="Filter: publishdate >= this (YYYY-MM-DD)."),
    date_to: Optional[str] = typer.Option(None, help="Filter: publishdate <= this (YYYY-MM-DD)."),
    min_sample_size: int = typer.Option(10, help="Refuse below this many comparable judgments."),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip the cross-encoder rerank step."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Bypass the persistent query cache."),
    verify: bool = typer.Option(False, "--verify", help="Run the structural faithfulness checker on the answer."),
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
        answer = observed_generate_answer(
            conn, index, embedder, category, situation,
            outcomes=outcome or None, date_from=date_from, date_to=date_to,
            min_sample_size=min_sample_size, reranker=reranker, use_cache=not no_cache,
        )

        if verify and not answer.refused:
            report = verify_answer(conn, answer, category)
            status = "[green]PASS[/green]" if report.passed else "[red]FAIL[/red]"
            console.print(
                f"[bold]Faithfulness check:[/bold] {status} "
                f"({report.checked_stats} stats, {report.checked_citations} citations checked)"
            )
            for issue in report.issues:
                console.print(f"  [red]-[/red] {issue}")
            console.print()

    if answer.refused:
        console.print(f"[bold red]Refused:[/bold red] {answer.refusal_reason}")
        raise typer.Exit(0)

    console.print(f"[bold]Sample size:[/bold] {answer.sample_size} comparable judgments")
    console.print(f"[bold]Outcome distribution:[/bold] {answer.outcome_distribution}\n")

    stats_table = Table(title="Compensation statistics (coverage = judgments with a non-null value)")
    stats_table.add_column("Field")
    stats_table.add_column("Coverage", justify="right")
    stats_table.add_column("Median", justify="right")
    stats_table.add_column("Min", justify="right")
    stats_table.add_column("Max", justify="right")

    def _row(stats):
        cov = f"{stats.coverage}/{stats.sample_size} ({stats.coverage_fraction*100:.0f}%)"
        stats_table.add_row(
            stats.field_name, cov,
            f"{stats.median:,.0f}" if stats.median is not None else "n/a",
            f"{stats.min:,.0f}" if stats.min is not None else "n/a",
            f"{stats.max:,.0f}" if stats.max is not None else "n/a",
        )

    if answer.amount_claimed_stats:
        _row(answer.amount_claimed_stats)
    for field_stats in answer.relief_component_stats.values():
        _row(field_stats)
    console.print(stats_table)

    console.print("\n[bold]Top cited judgments:[/bold]")
    for c in answer.citations:
        console.print(f"  tid={c.tid}  outcome={c.outcome}  relevance={c.relevance_score:.3f}")
        console.print(f"    {c.fact_summary[:150]}")


if __name__ == "__main__":
    app()
