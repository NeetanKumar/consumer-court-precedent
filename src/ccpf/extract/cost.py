"""LLM spend tracking, in integer micro-dollars (1e-6 USD).

Same discipline as ingest/budget.py's integer-paise ledger, for the same
reason: a float USD cap that drifts via repeated addition is a real way to
overspend against a metered API. Cumulative spend is recovered from
llm_spend_log at construction time, so a killed-and-restarted process
doesn't reset its notion of how much has already been spent.

Pricing verified against the Claude API skill (cached 2026-06-24). Sonnet 5
has an introductory rate ($2/$10 per MTok) through 2026-08-31; this uses the
standard rate ($3/$15) as a conservative (safe-overestimate) budget check.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

MICROUSD_PER_USD = 1_000_000

# USD per million tokens. Sonnet 5 uses its standard (non-introductory) rate
# deliberately, so the budget cap can't be exceeded by relying on a discount
# that expires 2026-08-31.
PRICING_PER_MTOK_USD = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "claude-fable-5": {"input": 10.00, "output": 50.00},  # independent validation reviewer
}


class LLMBudgetExceeded(RuntimeError):
    def __init__(self, spent_microusd: int, cap_microusd: int):
        super().__init__(
            f"LLM budget cap reached: spent ${spent_microusd / MICROUSD_PER_USD:.4f} "
            f"of ${cap_microusd / MICROUSD_PER_USD:.4f} cap."
        )
        self.spent_microusd = spent_microusd
        self.cap_microusd = cap_microusd


def estimate_cost_microusd(model: str, input_tokens: int, output_tokens: int) -> int:
    try:
        pricing = PRICING_PER_MTOK_USD[model]
    except KeyError:
        raise ValueError(f"Unknown model {model!r} — add it to PRICING_PER_MTOK_USD") from None
    usd = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
    return round(usd * MICROUSD_PER_USD)


class LLMBudgetTracker:
    def __init__(self, conn: sqlite3.Connection, cap_usd: float, run_id: str | None = None):
        self._conn = conn
        self.cap_microusd = round(cap_usd * MICROUSD_PER_USD)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        from ccpf.db import total_llm_spend_microusd

        self._spent_microusd = total_llm_spend_microusd(conn)

    @property
    def spent_microusd(self) -> int:
        return self._spent_microusd

    @property
    def spent_usd(self) -> float:
        return self._spent_microusd / MICROUSD_PER_USD

    def check(self) -> None:
        if self._spent_microusd >= self.cap_microusd:
            raise LLMBudgetExceeded(self._spent_microusd, self.cap_microusd)

    def record(self, model: str, ref: str, input_tokens: int, output_tokens: int) -> int:
        """Record actual spend from a completed API call. Returns cost in microusd.

        Recorded regardless of whether the response parsed successfully —
        Anthropic bills on the API call, not on our downstream validation.
        """
        cost = estimate_cost_microusd(model, input_tokens, output_tokens)
        self._conn.execute(
            "INSERT INTO llm_spend_log (ts, model, ref, input_tokens, output_tokens, cost_microusd, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                model,
                ref,
                input_tokens,
                output_tokens,
                cost,
                self.run_id,
            ),
        )
        self._conn.commit()
        self._spent_microusd += cost
        return cost
