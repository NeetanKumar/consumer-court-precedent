#!/usr/bin/env python3
"""Build the independent validation set: stratified sample + Opus 5 blind
review, stored in validation_labels. Runs independently of extraction.

Examples:
    python scripts/build_validation_set.py --category builder_delay --dry-run
    python scripts/build_validation_set.py --category builder_delay --sample-size 40
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import anthropic
import typer
from rich.console import Console
from rich.table import Table

from ccpf.config import DEFAULT_DB_PATH, get_settings
from ccpf.db import connection
from ccpf.extract.cost import LLMBudgetTracker, estimate_cost_microusd
from ccpf.extract.reviewer import REVIEWER_MODEL, IndependentReviewer
from ccpf.extract.sampling import stratified_sample
from ccpf.extract.validation_run import build_validation_set

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    category: str = typer.Option(..., help="Category key, e.g. builder_delay"),
    sample_size: int = typer.Option(40, help="Target validation set size."),
    medium_cap: int = typer.Option(15, help="Max medium-confidence docs to include."),
    seed: int = typer.Option(42, help="Sampling seed, for a reproducible set."),
    budget_usd: float = typer.Option(5.0, "--budget-usd", help="LLM budget cap for this run."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the sample and projected cost; make no API calls."),
):
    with connection(DEFAULT_DB_PATH) as conn:
        sample = stratified_sample(conn, category, sample_size=sample_size, medium_cap=medium_cap, seed=seed)

        conf_counts: dict[str, int] = {}
        outcome_counts: dict[str, int] = {}
        for row in sample:
            conf_counts[row.confidence] = conf_counts.get(row.confidence, 0) + 1
            outcome_counts[row.outcome] = outcome_counts.get(row.outcome, 0) + 1

        console.print(f"[cyan]Sample size:[/cyan] {len(sample)}")
        console.print(f"[cyan]By confidence:[/cyan] {conf_counts}")
        console.print(f"[cyan]By outcome:[/cyan] {outcome_counts}")

        if dry_run:
            # Rough projection: ~1500 input + ~350 output tokens/doc, typical for this prompt length.
            projected = estimate_cost_microusd(REVIEWER_MODEL, input_tokens=1500 * len(sample), output_tokens=350 * len(sample))
            console.print(f"[bold]Projected cost:[/bold] ${projected/1_000_000:.4f}")
            return

        settings = get_settings()
        api_key = settings.require_anthropic_key()
        client = anthropic.Anthropic(api_key=api_key)
        budget = LLMBudgetTracker(conn, cap_usd=budget_usd)
        reviewer = IndependentReviewer(client, budget)

        summary = build_validation_set(reviewer, conn, budget, category, sample)
        _print_summary(summary)


def _print_summary(summary) -> None:
    table = Table(title=f"Validation set build: {summary.category}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Attempted", str(summary.attempted))
    table.add_row("Succeeded", str(summary.succeeded))
    table.add_row("Skipped (already labeled)", str(summary.skipped_existing))
    table.add_row("Failed", str(summary.failed))
    table.add_row("Spend this run ($)", f"{summary.spend_this_run_usd:.4f}")
    table.add_row("Cumulative LLM spend ($)", f"{summary.spend_end_microusd/1_000_000:.4f}")
    console.print(table)
    if summary.stopped_early:
        console.print(f"[yellow]Stopped early:[/yellow] {summary.stopped_early}")
    for f in summary.failures[:10]:
        console.print(f"[red]Failure:[/red] {f}")


if __name__ == "__main__":
    app()
