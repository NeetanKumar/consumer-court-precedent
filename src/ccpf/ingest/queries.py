"""Category -> search query resolution.

Query text and ids live in config.yaml (see CategoryConfig), not hardcoded
here, so adding a query variant or tuning date ranges doesn't require a code
change. This module just exposes a typed accessor and documents the
convention query ids must follow.

Query ids are the checkpoint key in search_pages — changing a query's TEXT
without changing its ID would silently reuse stale pagination checkpoints
against a different query. If you edit query text in config.yaml, bump the id
(e.g. add a `_v2` suffix) so already-fetched pages aren't skipped incorrectly.
"""
from __future__ import annotations

from ccpf.config import CategoryConfig, QueryConfig


def queries_for_category(category: CategoryConfig) -> list[QueryConfig]:
    if not category.queries:
        raise ValueError("Category has no queries configured.")
    seen_ids = set()
    for q in category.queries:
        if q.id in seen_ids:
            raise ValueError(f"Duplicate query id {q.id!r} in category config.")
        seen_ids.add(q.id)
    return category.queries
