"""Retrieval: vector, keyword (BM25), hybrid (reciprocal rank fusion), optional query rewriting (HyDE +
keywords), optional cross-encoder rerank, and context packing (with small-to-big expansion)."""

from __future__ import annotations

import time
from dataclasses import replace

from . import config
from .embeddings import Embedder, get_embedder
from .models import Chunk, Hit, approx_tokens
from .store import Store

MODES = ("vector", "keyword", "hybrid")
DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Scores (query, chunk) pairs jointly. Slower than embeddings but much sharper at the top of the list.

    Options in fastembed: Xenova/ms-marco-MiniLM-L-6-v2 (80 MB, web-search trained),
    jinaai/jina-reranker-v1-turbo-en (150 MB), jinaai/jina-reranker-v2-base-multilingual (1.1 GB, knows code),
    BAAI/bge-reranker-base (1 GB).
    """

    def __init__(self, model: str = DEFAULT_RERANKER, max_chars: int = 2000):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = model
        self.max_chars = max_chars
        self._model = TextCrossEncoder(model_name=model, cache_dir=str(config.home() / "models"))

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        docs = [h.chunk.embedding_text()[: self.max_chars] for h in hits]
        scores = list(self._model.rerank(query, docs, batch_size=16))
        out = [replace(h, score=float(s), ranks={**h.ranks}) for h, s in zip(hits, scores)]
        out.sort(key=lambda h: -h.score)
        for i, h in enumerate(out, 1):
            h.ranks["rerank"] = i
        return out


def rrf(result_lists: dict[str, list[tuple[Chunk, float]]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion: score = sum over retrievers of 1 / (k + rank).

    Uses ranks only, so it doesn't care that BM25 scores and cosine similarities live on
    completely different scales.
    """
    fused: dict[str, Hit] = {}
    for source, results in result_lists.items():
        for rank, (chunk, _score) in enumerate(results, 1):
            hit = fused.setdefault(chunk.id, Hit(chunk=chunk, score=0.0))
            hit.score += 1.0 / (k + rank)
            hit.ranks[source] = rank
    return sorted(fused.values(), key=lambda h: -h.score)


class Retriever:
    def __init__(
        self,
        store: Store,
        mode: str = "hybrid",
        reranker: CrossEncoderReranker | None = None,
        candidates: int = 40,
        embedder: Embedder | None = None,
        rewriter=None,  # an LLM: enables query rewriting (keywords + hypothetical code)
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.store = store
        self.mode = mode
        self.reranker = reranker
        self.candidates = candidates
        self.rewriter = rewriter
        self._embedder = embedder
        self.last_trace: dict = {}
        self.last_lists: dict[str, list[tuple[Chunk, float]]] = {}

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = get_embedder(self.store.meta.get("embedder_spec", "fastembed"))
        return self._embedder

    def search(self, query: str, k: int = 8) -> list[Hit]:
        t0 = time.perf_counter()
        n = max(k, self.candidates)
        trace: dict = {"query": query, "mode": self.mode, "timings_ms": {}}
        plan = None
        if self.rewriter is not None:
            from .rewrite import plan_query

            plan = plan_query(query, self.rewriter)
            trace["rewrite"] = {"keywords": plan.keywords, "hypothetical_code": plan.hypothetical_code}
            trace["timings_ms"]["rewrite"] = round((time.perf_counter() - t0) * 1000)

        lists: dict[str, list[tuple[Chunk, float]]] = {}
        t1 = time.perf_counter()
        if self.mode in ("vector", "hybrid"):
            lists["vector"] = self.store.vector_search(self.embedder.embed_query(query), n)
            if plan and plan.hypothetical_code:  # HyDE: search with what the answer might look like
                lists["hyde"] = self.store.vector_search(self.embedder.embed_query(plan.hypothetical_code), n)
        if self.mode in ("keyword", "hybrid"):
            lists["keyword"] = self.store.keyword_search(plan.keyword_query if plan else query, n)
        trace["timings_ms"]["search"] = round((time.perf_counter() - t1) * 1000)

        if len(lists) > 1:
            hits = rrf(lists)
        else:
            (source, results), = lists.items()
            hits = [Hit(c, s, {source: i}) for i, (c, s) in enumerate(results, 1)]

        if self.reranker:
            t2 = time.perf_counter()
            hits = self.reranker.rerank(query, hits[:n])
            trace["timings_ms"]["rerank"] = round((time.perf_counter() - t2) * 1000)

        hits = hits[:k]
        trace["lists"] = {s: [c.citation for c, _ in r[:10]] for s, r in lists.items()}
        trace["final"] = [{"citation": h.chunk.citation, "symbol": h.chunk.symbol, "score": round(h.score, 4),
                           "ranks": h.ranks} for h in hits]
        trace["timings_ms"]["total"] = round((time.perf_counter() - t0) * 1000)
        self.last_trace, self.last_lists = trace, lists
        return hits


def pack_context(
    hits: list[Hit],
    budget_tokens: int,
    store: Store | None = None,
    expand: bool = False,
    max_parent_tokens: int = 1200,
) -> list[Chunk]:
    """Fit retrieved chunks into a token budget, in rank order.

    Without ``expand``: chunks already covered by a higher-ranked chunk are skipped and the chunk that
    crosses the budget is truncated by lines (this is where whole-file chunks lose the relevant code).

    With ``expand`` (small-to-big): each chunk is swapped for its parent span (the whole method, or the
    neighbouring windows) when that fits ``max_parent_tokens``, and spans from the same file that
    overlap or touch are merged, so the LLM sees one coherent block instead of fragments.
    """
    if expand and store is not None:
        return _pack_expanded(hits, budget_tokens, store, max_parent_tokens)
    packed: list[Chunk] = []
    used = 0
    for h in hits:
        c = h.chunk
        if any(p.path == c.path and p.start_line <= c.start_line and c.end_line <= p.end_line for p in packed):
            continue
        remaining = budget_tokens - used
        if remaining <= 20:
            break
        cost = approx_tokens(c.text)
        if cost > remaining:
            c = _truncate(c, remaining)
            if c is None:
                break
            cost = approx_tokens(c.text)
        packed.append(c)
        used += cost
    return packed


def _truncate(c: Chunk, tokens: int) -> Chunk | None:
    kept, t = [], 0
    for line in c.text.split("\n"):
        t += approx_tokens(line) if line.strip() else 0
        if t > tokens:
            break
        kept.append(line)
    if not kept:
        return None
    return replace(c, text="\n".join(kept), end_line=c.start_line + len(kept) - 1, id=c.id + "~")


def _span_tokens(lines: list[str], a: int, b: int) -> int:
    return sum(approx_tokens(l) for l in lines[a - 1 : b] if l.strip()) or 1


def _pack_expanded(hits: list[Hit], budget: int, store: Store, max_parent_tokens: int) -> list[Chunk]:
    spans: list[dict] = []  # {"chunk": first chunk, "a": int, "b": int, "lines": [...]}
    used = 0
    for h in hits:
        c = h.chunk
        lines = store.file_lines(c.path)
        if lines is None:  # old index without stored files: fall back to the chunk itself
            if used + approx_tokens(c.text) <= budget:
                spans.append({"chunk": c, "a": c.start_line, "b": c.end_line, "lines": None})
                used += approx_tokens(c.text)
            continue
        a, b = c.start_line, c.end_line
        if c.parent_start and _span_tokens(lines, c.parent_start, c.parent_end) <= max_parent_tokens:
            a, b = c.parent_start, c.parent_end
        same = [s for s in spans if s["chunk"].path == c.path and s["lines"] is not None
                and s["a"] - 3 <= b and a <= s["b"] + 3]
        if same:
            s = same[0]
            if s["a"] <= a and b <= s["b"]:
                continue  # already covered
            na, nb = min(s["a"], a), max(s["b"], b)
            delta = _span_tokens(lines, na, nb) - _span_tokens(lines, s["a"], s["b"])
            if used + delta <= budget:
                s["a"], s["b"] = na, nb
                used += delta
            continue
        cost = _span_tokens(lines, a, b)
        if used + cost > budget:
            a, b = c.start_line, c.end_line  # try the small chunk
            cost = _span_tokens(lines, a, b)
            if used + cost > budget:
                remaining = budget - used
                if remaining <= 20:
                    break
                while b > a and _span_tokens(lines, a, b) > remaining:
                    b -= 1
                cost = _span_tokens(lines, a, b)
                if cost > remaining:
                    break
        spans.append({"chunk": c, "a": a, "b": b, "lines": lines})
        used += cost
    out = []
    for s in spans:
        c = s["chunk"]
        if s["lines"] is None:
            out.append(c)
            continue
        out.append(replace(c, start_line=s["a"], end_line=s["b"], text="\n".join(s["lines"][s["a"] - 1 : s["b"]]),
                           id=f"{c.id}+{s['a']}-{s['b']}"))
    return out
