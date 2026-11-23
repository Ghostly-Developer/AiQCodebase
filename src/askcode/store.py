"""Vector stores.

* ``LocalStore``     numpy matrix + JSONL on disk, brute-force cosine search, in-memory BM25.
                     Zero setup; fine up to ~100k chunks.
* ``PgVectorStore``  Postgres + pgvector (HNSW index, cosine distance) with Postgres full-text
                     search for the keyword side. What you'd run in production.

Both expose the same calls: ``add``, ``replace_files``, ``vector_search``, ``keyword_search``,
``file_lines`` (full file text, for small-to-big expansion) and ``save_meta``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from . import config
from .bm25 import BM25, tokenize
from .models import Chunk


class Store:
    meta: dict

    def add(self, chunks: list[Chunk], vectors: np.ndarray, files: dict[str, str] | None = None) -> None:
        raise NotImplementedError

    def replace_files(self, remove_paths: set[str], chunks: list[Chunk], vectors: np.ndarray,
                      files: dict[str, str]) -> None:
        """Drop all chunks/files under remove_paths, then add the new ones (incremental update)."""
        raise NotImplementedError

    def vector_search(self, q: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        raise NotImplementedError

    def keyword_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        raise NotImplementedError

    def chunks(self) -> list[Chunk]:
        raise NotImplementedError

    def file_lines(self, path: str) -> list[str] | None:
        raise NotImplementedError

    def save_meta(self) -> None:
        raise NotImplementedError


def _keyword_text(c: Chunk) -> str:
    # index path, symbol and LLM summary too, so "SqlMapper.Async" style questions hit
    return f"{c.path} {c.symbol or ''} {c.context or ''}\n{c.text}"


class LocalStore(Store):
    def __init__(self, name: str):
        self.name = name
        self.dir: Path = config.index_dir(name)
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._bm25: BM25 | None = None
        self._files: dict[str, str] = {}
        self._lines: dict[str, list[str]] = {}
        self.meta = {}

    @classmethod
    def create(cls, name: str, meta: dict) -> LocalStore:
        s = cls(name)
        if s.dir.exists():
            shutil.rmtree(s.dir)
        s.dir.mkdir(parents=True)
        s.meta = meta
        return s

    @classmethod
    def open(cls, name: str) -> LocalStore:
        s = cls(name)
        if not (s.dir / "meta.json").exists():
            raise FileNotFoundError(f"No index named {name!r} in {s.dir.parent}. Run `askcode index` first.")
        s.meta = json.loads((s.dir / "meta.json").read_text(encoding="utf-8"))
        with open(s.dir / "chunks.jsonl", encoding="utf-8") as fh:
            s._chunks = [Chunk.from_dict(json.loads(line)) for line in fh]
        s._vectors = np.load(s.dir / "vectors.npy")
        if (s.dir / "files.json").exists():
            s._files = json.loads((s.dir / "files.json").read_text(encoding="utf-8"))
        return s

    def _save(self) -> None:
        with open(self.dir / "chunks.jsonl", "w", encoding="utf-8") as fh:
            for c in self._chunks:
                fh.write(json.dumps(c.to_dict()) + "\n")
        np.save(self.dir / "vectors.npy", self._vectors)
        (self.dir / "files.json").write_text(json.dumps(self._files), encoding="utf-8")
        self.save_meta()

    def save_meta(self) -> None:
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    def add(self, chunks: list[Chunk], vectors: np.ndarray, files: dict[str, str] | None = None) -> None:
        self._chunks.extend(chunks)
        self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
        self._files.update(files or {})
        self._bm25 = None
        self._lines.clear()
        self._save()

    def replace_files(self, remove_paths, chunks, vectors, files) -> None:
        keep = [i for i, c in enumerate(self._chunks) if c.path not in remove_paths]
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[keep] if self._vectors is not None else None
        for p in remove_paths:
            self._files.pop(p, None)
        if len(chunks):
            self.add(chunks, vectors, files)
        else:
            self._bm25 = None
            self._lines.clear()
            self._save()

    def vector_search(self, q: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        scores = self._vectors @ q
        top = np.argsort(-scores)[:k]
        return [(self._chunks[i], float(scores[i])) for i in top]

    def keyword_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            self._bm25 = BM25([_keyword_text(c) for c in self._chunks])
        return [(self._chunks[i], s) for i, s in self._bm25.search(query, k)]

    def chunks(self) -> list[Chunk]:
        return self._chunks

    def file_lines(self, path: str) -> list[str] | None:
        if path not in self._lines and path in self._files:
            self._lines[path] = self._files[path].split("\n")
        return self._lines.get(path)


class PgVectorStore(Store):
    """Postgres + pgvector. Start one with `docker compose up -d` and set DATABASE_URL."""

    def __init__(self, name: str, dsn: str | None = None):
        import psycopg
        from pgvector.psycopg import register_vector

        self.name = name
        self.dsn = dsn or config.env("DATABASE_URL", "postgresql://askcode:askcode@localhost:5432/askcode")
        self.conn = psycopg.connect(self.dsn, autocommit=True, connect_timeout=5)
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        self.conn.execute("CREATE TABLE IF NOT EXISTS askcode_indexes (name text PRIMARY KEY, meta jsonb NOT NULL)")
        self.meta = {}
        self._lines: dict[str, list[str]] = {}

    @property
    def table(self) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in self.name.lower())
        return f"askcode_chunks_{safe}"

    @property
    def files_table(self) -> str:
        return self.table.replace("askcode_chunks_", "askcode_files_", 1)

    @classmethod
    def create(cls, name: str, meta: dict, dsn: str | None = None) -> PgVectorStore:
        s = cls(name, dsn)
        s.meta = meta
        dim = int(meta["dim"])
        s.conn.execute(f"DROP TABLE IF EXISTS {s.table}")
        s.conn.execute(f"DROP TABLE IF EXISTS {s.files_table}")
        s.conn.execute(
            f"""CREATE TABLE {s.table} (
                id text PRIMARY KEY,
                path text NOT NULL,
                data jsonb NOT NULL,
                search_text text NOT NULL,
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', search_text)) STORED,
                embedding vector({dim}) NOT NULL
            )"""
        )
        s.conn.execute(f"CREATE TABLE {s.files_table} (path text PRIMARY KEY, body text NOT NULL)")
        s.save_meta()
        return s

    @classmethod
    def open(cls, name: str, dsn: str | None = None) -> PgVectorStore:
        s = cls(name, dsn)
        row = s.conn.execute("SELECT meta FROM askcode_indexes WHERE name = %s", (name,)).fetchone()
        if not row:
            raise FileNotFoundError(f"No pgvector index named {name!r}")
        s.meta = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        return s

    def save_meta(self) -> None:
        self.conn.execute(
            "INSERT INTO askcode_indexes (name, meta) VALUES (%s, %s) "
            "ON CONFLICT (name) DO UPDATE SET meta = EXCLUDED.meta",
            (self.name, json.dumps(self.meta)),
        )

    def add(self, chunks: list[Chunk], vectors: np.ndarray, files: dict[str, str] | None = None) -> None:
        with self.conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self.table} (id, path, data, search_text, embedding) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO NOTHING",
                [
                    (c.id, c.path, json.dumps(c.to_dict()), " ".join(tokenize(_keyword_text(c))), v)
                    for c, v in zip(chunks, vectors)
                ],
            )
            if files:
                cur.executemany(
                    f"INSERT INTO {self.files_table} (path, body) VALUES (%s, %s) "
                    "ON CONFLICT (path) DO UPDATE SET body = EXCLUDED.body",
                    list(files.items()),
                )
        # HNSW: approximate nearest-neighbour index; GIN for full-text. IF NOT EXISTS keeps updates cheap.
        self.conn.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_hnsw ON {self.table} USING hnsw (embedding vector_cosine_ops)"
        )
        self.conn.execute(f"CREATE INDEX IF NOT EXISTS {self.table}_tsv ON {self.table} USING gin (tsv)")
        self._lines.clear()
        self.save_meta()

    def replace_files(self, remove_paths, chunks, vectors, files) -> None:
        if remove_paths:
            self.conn.execute(f"DELETE FROM {self.table} WHERE path = ANY(%s)", (list(remove_paths),))
            self.conn.execute(f"DELETE FROM {self.files_table} WHERE path = ANY(%s)", (list(remove_paths),))
        self.add(chunks, vectors, files)

    def vector_search(self, q: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        rows = self.conn.execute(
            f"SELECT data, 1 - (embedding <=> %s) AS score FROM {self.table} ORDER BY embedding <=> %s LIMIT %s",
            (q, q, k),
        ).fetchall()
        return [(Chunk.from_dict(d), float(s)) for d, s in rows]

    def keyword_search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        terms = sorted(set(tokenize(query)))
        if not terms:
            return []
        tsquery = " | ".join(t.replace("'", "") for t in terms)
        rows = self.conn.execute(
            f"SELECT data, ts_rank_cd(tsv, q) AS score FROM {self.table}, to_tsquery('simple', %s) q "
            "WHERE tsv @@ q ORDER BY score DESC LIMIT %s",
            (tsquery, k),
        ).fetchall()
        return [(Chunk.from_dict(d), float(s)) for d, s in rows]

    def chunks(self) -> list[Chunk]:
        return [Chunk.from_dict(d) for (d,) in self.conn.execute(f"SELECT data FROM {self.table}").fetchall()]

    def file_lines(self, path: str) -> list[str] | None:
        if path not in self._lines:
            row = self.conn.execute(f"SELECT body FROM {self.files_table} WHERE path = %s", (path,)).fetchone()
            if not row:
                return None
            self._lines[path] = row[0].split("\n")
        return self._lines[path]


def create_store(kind: str, name: str, meta: dict) -> Store:
    if kind == "local":
        return LocalStore.create(name, meta)
    if kind == "pgvector":
        return PgVectorStore.create(name, meta)
    raise ValueError(f"Unknown store {kind!r}")


def open_store(name: str, kind: str | None = None) -> Store:
    """Open an index by name. Local indexes are found on disk; otherwise try pgvector."""
    if kind == "pgvector":
        return PgVectorStore.open(name)
    try:
        return LocalStore.open(name)
    except FileNotFoundError:
        if kind == "local" or not config.env("DATABASE_URL"):
            raise
        return PgVectorStore.open(name)


def list_local() -> list[dict]:
    root = config.home() / "indexes"
    out = []
    if root.exists():
        for d in sorted(root.iterdir()):
            if (d / "meta.json").exists():
                out.append({"name": d.name, **json.loads((d / "meta.json").read_text(encoding="utf-8"))})
    return out
