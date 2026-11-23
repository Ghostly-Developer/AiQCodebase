"""Embedding providers. All return L2-normalised float32 matrices so cosine similarity == dot product."""

from __future__ import annotations

import hashlib
import re
import time

import numpy as np

from . import config


def _normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.clip(norms, 1e-12, None)


class Embedder:
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class FastEmbedEmbedder(Embedder):
    """Local ONNX model via fastembed. Free, offline after the first download (~130 MB for bge-small).

    Note: bge-small truncates input at 512 tokens, so a whole-file chunk is embedded from its
    first ~2,000 characters only. That is one reason file-level chunking retrieves badly.
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", max_chars: int = 8000):
        from fastembed import TextEmbedding

        self.name = f"fastembed:{model}"
        cache = config.home() / "models"
        self._model = TextEmbedding(model_name=model, cache_dir=str(cache))
        self.dim = len(next(iter(self._model.query_embed(["probe"]))))
        # long-context models (jina 8k) would otherwise spend minutes on whole files; the
        # model can't use more than its context anyway
        self._max_chars = max_chars

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        texts = [t[: self._max_chars] for t in texts]
        return _normalize(np.stack(list(self._model.passage_embed(texts, batch_size=16))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.stack(list(self._model.query_embed([text]))))[0]


class GoogleEmbedder(Embedder):
    """Gemini embeddings (needs GOOGLE_API_KEY). Uses asymmetric task types for docs vs queries."""

    def __init__(self, model: str | None = None, dim: int = 768):
        from google import genai

        key = config.env("GOOGLE_API_KEY") or config.env("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set (needed for --embedder google)")
        self._client = genai.Client(api_key=key)
        self._model = model or config.env("ASKCODE_GOOGLE_EMBED_MODEL", "gemini-embedding-001")
        self.name = f"google:{self._model}:{dim}"
        self.dim = dim

    def _embed(self, texts: list[str], task: str) -> np.ndarray:
        from google.genai import types

        out: list[list[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            for attempt in range(5):
                try:
                    resp = self._client.models.embed_content(
                        model=self._model,
                        contents=batch,
                        config=types.EmbedContentConfig(task_type=task, output_dimensionality=self.dim),
                    )
                    break
                except Exception as e:  # rate limits on the free tier are common
                    if attempt == 4 or "429" not in str(e):
                        raise
                    time.sleep(2**attempt * 5)
            out.extend(e.values for e in resp.embeddings)
        return _normalize(np.array(out))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "CODE_RETRIEVAL_QUERY")[0]


class HashEmbedder(Embedder):
    """Deterministic bag-of-words hashing embedder. No downloads; used by the unit tests."""

    def __init__(self, dim: int = 256):
        self.name = f"hash:{dim}"
        self.dim = dim

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._one(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class CachedEmbedder(Embedder):
    """Disk cache keyed on (model, text): re-indexing unchanged chunks costs nothing."""

    def __init__(self, inner: Embedder):
        from .cache import KVCache

        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self._cache = KVCache("embeddings")

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        from .cache import KVCache

        keys = [KVCache.key(self.name, "doc", t) for t in texts]
        found = self._cache.get_many(keys)
        missing = [i for i, k in enumerate(keys) if k not in found]
        if missing:
            new = self.inner.embed_documents([texts[i] for i in missing])
            self._cache.set_many([(keys[i], new[j].astype(np.float32).tobytes()) for j, i in enumerate(missing)])
            for j, i in enumerate(missing):
                found[keys[i]] = new[j].astype(np.float32).tobytes()
        return np.stack([np.frombuffer(found[k], dtype=np.float32) for k in keys])

    def embed_query(self, text: str) -> np.ndarray:
        return self.inner.embed_query(text)


def get_embedder(spec: str, cache: bool = True) -> Embedder:
    """spec: 'fastembed', 'fastembed:<model>', 'google', 'google:<model>', or 'hash'."""
    kind, _, model = spec.partition(":")
    if kind == "fastembed":
        emb: Embedder = FastEmbedEmbedder(model) if model else FastEmbedEmbedder()
    elif kind == "google":
        emb = GoogleEmbedder(model or None)
    elif kind == "hash":
        return HashEmbedder()
    else:
        raise ValueError(f"Unknown embedder {spec!r} (use fastembed, google or hash)")
    return CachedEmbedder(emb) if cache else emb
