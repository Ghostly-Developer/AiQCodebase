from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import loader
from .chunking import chunker_from_meta, get_chunker
from .embeddings import get_embedder
from .models import approx_tokens
from .store import Store, create_store, open_store


@dataclass
class IndexStats:
    name: str
    files: int
    chunks: int
    avg_tokens: float
    max_tokens: int
    seconds: float
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    contextualized: int = 0
    redactions: int = 0


def build_index(
    name: str,
    source: str,
    chunker: str = "function",
    embedder: str = "fastembed",
    store: str = "local",
    ref: str | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    contextualize_llm=None,
    progress=None,
) -> tuple[Store, IndexStats]:
    t0 = time.time()
    repo = loader.load(source, ref=ref, include=include, exclude=exclude)
    ch = get_chunker(chunker)
    chunks = ch.chunk_all(repo.files)
    if not chunks:
        raise ValueError(f"No indexable files found in {repo.root}")
    n_ctx = 0
    if contextualize_llm is not None:
        from .contextualize import contextualize

        if progress:
            progress(f"Summarising {len(chunks)} chunks with {contextualize_llm.name} (cached after the first run)")
        n_ctx = contextualize(chunks, contextualize_llm)
    emb = get_embedder(embedder)
    if progress:
        progress(f"Embedding {len(chunks)} chunks from {len(repo.files)} files with {emb.name}")
    vectors = emb.embed_documents([c.embedding_text() for c in chunks])

    tokens = [approx_tokens(c.text) for c in chunks]
    meta = {
        "source": repo.source,
        "ref": ref,
        "root": str(repo.root),
        "commit": repo.commit,
        "web_base": repo.web_base,
        "chunker": ch.describe(),
        "embedder": emb.name,
        "embedder_spec": embedder,
        "dim": emb.dim,
        "store": store,
        "files": len(repo.files),
        "chunks": len(chunks),
        "include": include,
        "exclude": exclude,
        "contextualized": contextualize_llm.name if contextualize_llm is not None else None,
        "file_hashes": {f.path: f.sha for f in repo.files},
        "redactions": repo.redactions,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    st = create_store(store, name, meta)
    st.add(chunks, vectors, {f.path: f.text for f in repo.files})
    stats = IndexStats(
        name=name, files=len(repo.files), chunks=len(chunks), avg_tokens=sum(tokens) / len(tokens),
        max_tokens=max(tokens), seconds=time.time() - t0, contextualized=n_ctx, redactions=repo.redactions,
    )
    return st, stats


def update_index(name: str, contextualize_llm=None, progress=None) -> tuple[Store, IndexStats]:
    """Re-index only files whose content changed since the last run (by content hash).

    Works for local paths (edit files, then update) and for git URLs indexed without a pinned
    ref (fetches the latest HEAD). Unchanged chunks keep their vectors; changed files are
    re-chunked and re-embedded (the embedding cache makes even that cheap for untouched chunks).
    """
    t0 = time.time()
    st = open_store(name)
    meta = st.meta
    repo = loader.load(meta["source"], ref=meta.get("ref"), include=meta.get("include"), exclude=meta.get("exclude"))
    old: dict[str, str] = meta.get("file_hashes", {})
    new = {f.path: f.sha for f in repo.files}
    changed = sorted(p for p, h in new.items() if old.get(p) != h)
    removed = sorted(set(old) - set(new))
    n_ctx = 0
    chunks = []
    if changed or removed:
        ch = chunker_from_meta(meta["chunker"])
        changed_files = [f for f in repo.files if f.path in set(changed)]
        chunks = ch.chunk_all(changed_files)
        if contextualize_llm is not None or meta.get("contextualized"):
            if contextualize_llm is not None:
                from .contextualize import contextualize

                n_ctx = contextualize(chunks, contextualize_llm)
        emb = get_embedder(meta["embedder_spec"])
        if progress:
            progress(f"{len(changed)} changed, {len(removed)} removed file(s); embedding {len(chunks)} chunks")
        import numpy as np

        vectors = emb.embed_documents([c.embedding_text() for c in chunks]) if chunks else np.zeros((0, meta["dim"]))
        meta["file_hashes"] = new
        meta["commit"] = repo.commit
        meta["web_base"] = repo.web_base
        meta["files"] = len(repo.files)
        meta["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        st.replace_files(set(changed) | set(removed), chunks, vectors, {f.path: f.text for f in changed_files})
        meta["chunks"] = len(st.chunks())
        st.save_meta()
    all_chunks = st.chunks()
    tokens = [approx_tokens(c.text) for c in all_chunks] or [0]
    return st, IndexStats(
        name=name, files=len(repo.files), chunks=len(all_chunks), avg_tokens=sum(tokens) / len(tokens),
        max_tokens=max(tokens), seconds=time.time() - t0, changed=changed, removed=removed, contextualized=n_ctx,
    )
