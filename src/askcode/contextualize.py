"""Contextual retrieval: an LLM writes one sentence per chunk describing its role in the file.

The sentence is prepended to the chunk before embedding and added to the keyword index, so a
question phrased in plain English ("how is the timeout chosen?") can match code that never uses
those words. Results are cached in the LLM disk cache, so re-indexing is free.
"""

from __future__ import annotations

import json
from collections import defaultdict

from .llm import LLM
from .models import Chunk

SYSTEM = (
    "You help index a code repository for search. For each numbered chunk from the same file, write ONE "
    "sentence (max 30 words) saying what the code does and why someone would look for it, using plain "
    "words a developer might search with. Mention key identifiers. Do not invent behaviour. "
    'Return JSON: {"summaries": ["...", "..."]} with exactly one string per chunk, in order.'
)


def contextualize(chunks: list[Chunk], llm: LLM, batch: int = 8, progress=None) -> int:
    """Fill chunk.context in place. Returns how many chunks got a summary."""
    by_file: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_file[c.path].append(c)
    done = 0
    total = len(chunks)
    for path, items in by_file.items():
        for i in range(0, len(items), batch):
            group = items[i : i + batch]
            body = "\n\n".join(
                f"### Chunk {n} (lines {c.start_line}-{c.end_line}{', ' + c.symbol if c.symbol else ''})\n{c.text[:3000]}"
                for n, c in enumerate(group, 1)
            )
            try:
                data = json.loads(llm.complete(SYSTEM, f"File: {path}\n\n{body}", json_mode=True))
                summaries = data.get("summaries", []) if isinstance(data, dict) else []
            except (json.JSONDecodeError, RuntimeError):
                summaries = []
            for c, s in zip(group, summaries):
                if isinstance(s, str) and s.strip():
                    c.context = s.strip()[:300]
                    done += 1
            if progress:
                progress(min(total, done), total)
    return done
