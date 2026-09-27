#!/usr/bin/env python3
"""CLI entry point for Stage 1 ingestion. Runs independently of the other stages.

Examples:
    python scripts/ingest.py --category builder_delay --dry-run
    python scripts/ingest.py --category builder_delay --max-pages 1 --max-docs 10
    python scripts/ingest.py --category builder_delay --budget-inr 400
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from rich.console import Console
from rich.table import Table

from ccpf.config import DEFAULT_CONFIG_PATH, DEFAULT_DB_PATH, DEFAULT_RAW_CACHE_DIR, get_settings, load_app_config
from ccpf.db import connection
from ccpf.ingest.budget import BudgetTracker
from ccpf.ingest.cache import RawCache
from ccpf.ingest.client import IndianKanoonClient
from ccpf.ingest.run import dry_run_projection, run_ingestion

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    category: str = typer.Option(..., help="Category key from config.yaml, e.g. builder_delay"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Project cost from page-0 `found`; fetch no documents."),
    max_pages: int = typer.Option(None, help="Cap search pages fetched per query this run."),
    max_docs: int = typer.Option(None, help="Cap documents fetched this run."),
    budget_inr: float = typer.Option(None, "--budget-inr", help="Override the category's configured budget cap."),
):
    app_config = load_app_config(DEFAULT_CONFIG_PATH)
    cat_config = app_config.get_category(category)
    settings = get_settings()
    token = settings.require_indiankanoon_token()

    cap_inr = budget_inr if budget_inr is not None else (cat_config.budget_inr or 0.0)
    if cap_inr <= 0:
        console.print("[red]No budget cap configured — pass --budget-inr explicitly.[/red]")
        raise typer.Exit(1)

    with connection(DEFAULT_DB_PATH) as conn:
        budget = BudgetTracker(conn, cap_inr, app_config.pricing_paise.model_dump())
        console.print(
            f"[cyan]Category:[/cyan] {category}  "
            f"[cyan]Cap:[/cyan] Rs.{cap_inr:.2f}  "
            f"[cyan]Already spent:[/cyan] Rs.{budget.spent_inr:.2f}"
        )

        with IndianKanoonClient(
            token=token,
            base_url=app_config.api.base_url,
            rate_limit_per_sec=app_config.api.rate_limit_per_sec,
            max_retries=app_config.api.max_retries,
            retry_min_wait_s=app_config.api.retry_min_wait_s,
            retry_max_wait_s=app_config.api.retry_max_wait_s,
        ) as client:
            if dry_run:
                projection = dry_run_projection(client, conn, category, cat_config, budget)
                _print_dry_run(projection)
                return

            cache = RawCache(DEFAULT_RAW_CACHE_DIR)
            summary = run_ingestion(
                client, conn, cache, category, cat_config, budget,
                max_pages=max_pages, max_docs=max_docs,
            )
            _print_summary(summary)


def _print_dry_run(projection: dict) -> None:
    table = Table(title="Dry-run cost projection (page-0 search cost already incurred)")
    table.add_column("Query ID")
    table.add_column("Found", justify="right")
    table.add_column("Projected search Rs.", justify="right")
    table.add_column("Projected doc Rs.", justify="right")
    for qid, p in projection["queries"].items():
        table.add_row(
            qid, str(p["found"]),
            f"{p['projected_search_paise']/100:.2f}",
            f"{p['projected_doc_paise']/100:.2f}",
        )
    console.print(table)
    console.print(f"[bold]Total projected additional spend:[/bold] Rs.{projection['total_projected_inr']:.2f}")
    console.print(f"[bold]Cap:[/bold] Rs.{projection['cap_inr']:.2f}  [bold]Already spent:[/bold] Rs.{projection['already_spent_inr']:.2f}")


def _print_summary(summary) -> None:
    table = Table(title=f"Ingestion summary: {summary.category}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Search pages fetched (new)", str(summary.search_pages_fetched))
    table.add_row("Docs fetched (new, paid)", str(summary.docs_fetched))
    table.add_row("Docs loaded from disk cache", str(summary.docs_from_cache))
    table.add_row("Docs already in DB (skipped)", str(summary.docs_skipped_existing))
    table.add_row("NCDRC-passed", str(summary.ncdrc_passed))
    table.add_row("Award-language-passed", str(summary.award_passed))
    table.add_row("Both filters passed", str(summary.both_passed))
    table.add_row("Spend this run (Rs.)", f"{summary.spend_this_run_inr:.2f}")
    table.add_row("Cumulative spend (Rs.)", f"{summary.spend_paise_end/100:.2f}")
    console.print(table)
    if summary.stopped_early:
        console.print(f"[yellow]Stopped early:[/yellow] {summary.stopped_early}")


if __name__ == "__main__":
    app()
