"""Integration test for the Postgres + pgvector backend. Runs only when DATABASE_URL is set (CI does this)."""

import os

import pytest

from askcode.indexer import build_index, update_index
from askcode.retrieval import Retriever, pack_context
from askcode.store import open_store

pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="DATABASE_URL not set")


def test_pgvector_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ASKCODE_HOME", str(tmp_path / "home"))
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "cache.py").write_text(
        "def purge_query_cache():\n    '''Clear the query cache.'''\n    CACHE.clear()\n\n\n"
        "def open_connection(dsn):\n    return connect(dsn)\n",
        encoding="utf-8",
    )
    store, stats = build_index("pgtest", str(repo), "function", "hash", "pgvector")
    assert stats.chunks >= 1

    store = open_store("pgtest", "pgvector")
    assert store.meta["chunker"]["name"] == "function"
    for mode in ("vector", "keyword", "hybrid"):
        hits = Retriever(store, mode=mode).search("purge the query cache", k=3)
        assert hits and hits[0].chunk.path == "cache.py"
    packed = pack_context(Retriever(store).search("purge cache", 3), 1000, store, expand=True)
    assert packed and store.file_lines("cache.py")[0].startswith("def purge_query_cache")

    (repo / "cache.py").write_text("def purge_everything():\n    pass\n", encoding="utf-8")
    store, upd = update_index("pgtest")
    assert upd.changed == ["cache.py"]
    assert all("purge_query_cache" not in c.text for c in store.chunks())
