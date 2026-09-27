"""Raw API response cache on disk, keyed by tid.

Separate from the SQLite raw_docs table on purpose: SQLite holds the parsed
record used by the rest of the pipeline, while this is an untouched copy of
exactly what the API returned. If a future parsing bug corrupts plain_text
extraction, the fix is a local reprocessing pass over these files — never a
re-fetch, which would cost real money again.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class RawCache:
    def __init__(self, cache_dir: Path):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, tid: int) -> Path:
        return self._dir / f"{tid}.json"

    def has(self, tid: int) -> bool:
        return self._path(tid).exists()

    def load(self, tid: int) -> Optional[dict]:
        path = self._path(tid)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def save(self, tid: int, data: dict) -> None:
        path = self._path(tid)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        tmp.replace(path)  # atomic-ish: avoid a half-written cache file on crash
