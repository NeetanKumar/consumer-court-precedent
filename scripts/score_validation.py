#!/usr/bin/env python3
"""Score Stage 2 extraction against the independent (Opus 5) gold labels.

Read-only — never touches the API. Run after build_validation_set.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from rich.console import Console
from rich.table import Table

from ccpf.config import DEFAULT_DB_PATH
from ccpf.db import connection
from ccpf.extract.scoring import score_validation_set

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(category: str = typer.Option(..., help="Category key, e.g. builder_delay")):
    with connection(DEFAULT_DB_PATH) as conn:
        report = score_validation_set(conn, category)

    if report.n == 0:
        console.print("[yellow]No validation labels found — run build_validation_set.py first.[/yellow]")
        raise typer.Exit(1)

    console.print(f"[bold]Validation set size:[/bold] {report.n}\n")

    table = Table(title="Field accuracy vs. independent Opus 5 gold labels")
    table.add_column("Field")
    table.add_column("Accuracy", justify="right")
    table.add_column("Matched", justify="right")
    table.add_column("Mismatched", justify="right")
    table.add_column("Gold-only (extraction missed)", justify="right")
    table.add_column("Extraction-only (false positive)", justify="right")
    table.add_column("Both null", justify="right")

    def _row(name, score):
        table.add_row(
            name,
            f"{score.accuracy*100:.0f}%" if score.total_comparable else "n/a (no comparable values)",
            str(score.matched), str(score.mismatched),
            str(score.gold_only), str(score.extraction_only), str(score.both_null),
        )

    _row("outcome", report.outcome)
    for path, score in report.numeric.items():
        _row(path, score)

    console.print(table)

    console.print("\n[bold]Outcome agreement rate by extraction confidence tier:[/bold]")
    for conf, rate in sorted(report.accuracy_by_confidence.items()):
        console.print(f"  {conf}: {rate*100:.0f}%")

    if report.n < 30:
        console.print(
            "\n[yellow]Validation set is below the ~30-50 doc target — "
            "widen the sample before treating these numbers as reliable.[/yellow]"
        )


if __name__ == "__main__":
    app()
