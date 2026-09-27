#!/usr/bin/env python3
"""CLI entry point for Stage 2 extraction. Runs independently of ingestion.

Examples:
    python scripts/extract.py --category builder_delay --max-docs 5
    python scripts/extract.py --category builder_delay --budget-usd 10

    # Re-test a prompt change against a known subset before a full re-run:
    python scripts/extract.py --category builder_delay --tids 135225184,149729084 --force
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import anthropic
import typer
from rich.console import Console
from rich.table import Table

from ccpf.config import DEFAULT_CONFIG_PATH, DEFAULT_DB_PATH, get_settings, load_app_config
from ccpf.db import connection
from ccpf.extract.cost import LLMBudgetTracker
from ccpf.extract.extractor import ClaudeExtractor
from ccpf.extract.run import run_extraction

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    category: str = typer.Option(..., help="Category key from config.yaml, e.g. builder_delay"),
    max_docs: int = typer.Option(None, help="Cap documents extracted this run."),
    budget_usd: float = typer.Option(None, "--budget-usd", help="Override the configured LLM budget cap."),
    tids: str = typer.Option(None, help="Comma-separated tids to restrict this run to, e.g. for testing a prompt change."),
    force: bool = typer.Option(False, "--force", help="With --tids, delete and re-extract docs that already have an extraction row."),
):
    app_config = load_app_config(DEFAULT_CONFIG_PATH)
    settings = get_settings()
    api_key = settings.require_anthropic_key()

    cap_usd = budget_usd if budget_usd is not None else app_config.extraction.budget_usd
    tid_list = [int(t.strip()) for t in tids.split(",")] if tids else None

    with connection(DEFAULT_DB_PATH) as conn:
        budget = LLMBudgetTracker(conn, cap_usd=cap_usd)
        console.print(
            f"[cyan]Category:[/cyan] {category}  "
            f"[cyan]LLM budget cap:[/cyan] ${cap_usd:.2f}  "
            f"[cyan]Already spent:[/cyan] ${budget.spent_usd:.4f}"
        )

        client = anthropic.Anthropic(api_key=api_key)
        extractor = ClaudeExtractor(client, budget)

        summary = run_extraction(extractor, conn, budget, category, max_docs=max_docs, tids=tid_list, force=force)
        _print_summary(summary)


def _print_summary(summary) -> None:
    table = Table(title=f"Extraction summary: {summary.category}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Attempted", str(summary.attempted))
    table.add_row("Succeeded", str(summary.succeeded))
    table.add_row("Escalated to Sonnet", str(summary.escalated))
    table.add_row("Failed (both models)", str(summary.failed))
    table.add_row("Spend this run ($)", f"{summary.spend_this_run_usd:.4f}")
    table.add_row("Cumulative LLM spend ($)", f"{summary.spend_end_microusd/1_000_000:.4f}")
    console.print(table)
    if summary.stopped_early:
        console.print(f"[yellow]Stopped early:[/yellow] {summary.stopped_early}")
    for f in summary.failures[:10]:
        console.print(f"[red]Failure:[/red] {f}")


if __name__ == "__main__":
    app()
