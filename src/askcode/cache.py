"""Tiny SQLite key-value cache for LLM responses and embeddings.

Eval re-runs hit the cache instead of the API, so they are free, fast and reproducible, and
free-tier rate limits stop mattering after the first run.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading

from . import config


class KVCache:
    def __init__(self, name: str):
        path = config.home() / "cache"
        path.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path / f"{name}.sqlite", check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB)")
        self._lock = threading.Lock()

    @staticmethod
    def key(*parts: str) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    def get(self, k: str):
        with self._lock:
            row = self._db.execute("SELECT v FROM kv WHERE k = ?", (k,)).fetchone()
        return row[0] if row else None

    def get_many(self, keys: list[str]) -> dict:
        out = {}
        with self._lock:
            for i in range(0, len(keys), 500):
                part = keys[i : i + 500]
                q = f"SELECT k, v FROM kv WHERE k IN ({','.join('?' * len(part))})"
                out.update(dict(self._db.execute(q, part).fetchall()))
        return out

    def set(self, k: str, v) -> None:
        self.set_many([(k, v)])

    def set_many(self, items: list[tuple[str, object]]) -> None:
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", items)
            self._db.commit()
