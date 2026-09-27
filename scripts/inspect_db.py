#!/usr/bin/env python3
"""Corpus stats: fetch counts, filter pass rates, date spread, total spend.

Read-only — never touches the API. This is the Stage-1 exit gate: check
counts and pass rates here before starting Stage 2 extraction.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from rich.console import Console
from rich.table import Table

from ccpf.config import DEFAULT_DB_PATH
from ccpf.db import connection, total_spend_paise

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(category: str = typer.Option(None, help="Restrict filter stats to one category.")):
    with connection(DEFAULT_DB_PATH) as conn:
        n_hits = conn.execute("SELECT COUNT(DISTINCT tid) AS n FROM search_hits").fetchone()["n"]
        n_docs = conn.execute("SELECT COUNT(*) AS n FROM raw_docs").fetchone()["n"]
        date_row = conn.execute(
            "SELECT MIN(publishdate) AS lo, MAX(publishdate) AS hi FROM raw_docs WHERE publishdate != ''"
        ).fetchone()
        spend = total_spend_paise(conn)

        console.print(f"[bold]Search hits (deduped tids):[/bold] {n_hits}")
        console.print(f"[bold]Full docs fetched:[/bold] {n_docs}")
        console.print(f"[bold]Publish date range:[/bold] {date_row['lo']} .. {date_row['hi']}")
        console.print(f"[bold]Cumulative spend:[/bold] Rs.{spend/100:.2f}")

        where = "WHERE category = ?" if category else ""
        params = (category,) if category else ()
        rows = conn.execute(
            f"""
            SELECT category,
                   COUNT(*) AS total,
                   SUM(is_ncdrc) AS ncdrc_pass,
                   SUM(has_award_language) AS award_pass,
                   SUM(passed) AS both_pass
            FROM doc_filters
            {where}
            GROUP BY category
            """,
            params,
        ).fetchall()

        table = Table(title="Filter pass rates by category")
        table.add_column("Category")
        table.add_column("Total", justify="right")
        table.add_column("NCDRC pass", justify="right")
        table.add_column("Award-language pass", justify="right")
        table.add_column("Both pass", justify="right")
        for row in rows:
            table.add_row(
                row["category"], str(row["total"]),
                f"{row['ncdrc_pass']} ({row['ncdrc_pass']/row['total']*100:.0f}%)",
                f"{row['award_pass']} ({row['award_pass']/row['total']*100:.0f}%)",
                f"{row['both_pass']} ({row['both_pass']/row['total']*100:.0f}%)",
            )
        console.print(table)

        if rows and rows[0]["both_pass"] < 150:
            console.print(
                "[yellow]Below the ~150-doc gate suggested before starting Stage 2. "
                "Consider widening queries or filter patterns (both free to re-run).[/yellow]"
            )


if __name__ == "__main__":
    app()
