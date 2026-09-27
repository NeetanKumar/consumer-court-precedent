"""Spend cap enforcement against the paid Indian Kanoon API.

The tracker recovers cumulative spend from spend_log at construction time,
so a killed-and-restarted process doesn't reset its notion of how much has
already been spent. check_and_reserve() must be called BEFORE every paid
request; it raises BudgetExceeded instead of letting the caller find out
after the fact.
"""
from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

from ccpf import db


class BudgetExceeded(RuntimeError):
    def __init__(self, endpoint: str, spent_paise: int, cap_paise: int):
        super().__init__(
            f"Budget cap reached: spent {spent_paise / 100:.2f} INR of "
            f"{cap_paise / 100:.2f} INR cap; next '{endpoint}' call would exceed it."
        )
        self.endpoint = endpoint
        self.spent_paise = spent_paise
        self.cap_paise = cap_paise


class BudgetTracker:
    def __init__(
        self,
        conn: sqlite3.Connection,
        cap_inr: float,
        pricing_paise: dict[str, int],
        run_id: str | None = None,
    ):
        self._conn = conn
        self.cap_paise = round(cap_inr * 100)
        self._pricing_paise = pricing_paise
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._spent_paise = db.total_spend_paise(conn)

    @property
    def spent_paise(self) -> int:
        return self._spent_paise

    @property
    def spent_inr(self) -> float:
        return self._spent_paise / 100

    @property
    def remaining_paise(self) -> int:
        return max(0, self.cap_paise - self._spent_paise)

    def cost_of(self, endpoint: str) -> int:
        try:
            return self._pricing_paise[endpoint]
        except KeyError:
            raise ValueError(f"Unknown endpoint {endpoint!r} in pricing config") from None

    def check_and_reserve(self, endpoint: str, ref: str = "") -> int:
        """Raise BudgetExceeded if the next call would exceed the cap.

        Does NOT record spend yet — call record_spend() only after the API
        call actually succeeds, so a failed request doesn't get charged.
        Returns the cost in paise for convenience.
        """
        cost = self.cost_of(endpoint)
        if self._spent_paise + cost > self.cap_paise:
            raise BudgetExceeded(endpoint, self._spent_paise, self.cap_paise)
        return cost

    def record_spend(self, endpoint: str, cost_paise: int, ref: str = "") -> None:
        self._conn.execute(
            "INSERT INTO spend_log (ts, endpoint, ref, cost_paise, run_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), endpoint, ref, cost_paise, self.run_id),
        )
        self._conn.commit()
        self._spent_paise += cost_paise

    def project_search_cost(self, found: int, page_size: int = 10) -> int:
        """Rough dry-run projection: pages needed to see `found` results."""
        import math

        pages = max(1, math.ceil(found / page_size)) if found else 0
        return pages * self.cost_of("search")
