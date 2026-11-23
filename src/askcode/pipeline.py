"""The full question-answering pipeline, shared by the CLI (`ask`, `chat`) and the web server.

    follow-up condensation -> retrieval (hybrid, optional rewrite/rerank) -> context packing
    (optional small-to-big) -> grounded answer (streamed) -> citation verification -> trace log
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from .answer import Answer, finalize, stream_answer
from .llm import LLM, get_llm
from .models import Chunk
from .retrieval import DEFAULT_RERANKER, CrossEncoderReranker, Retriever, pack_context
from .rewrite import condense
from .store import Store, open_store


@dataclass
class AskOptions:
    mode: str = "hybrid"
    rerank: bool = False
    reranker_model: str = DEFAULT_RERANKER
    rewrite: bool = False
    expand: bool = True  # small-to-big
    k: int = 8
    budget: int = 3000


class Assistant:
    def __init__(self, index: str, llm: str | LLM | None = None, opts: AskOptions | None = None,
                 trace_path: Path | None = None, store: Store | None = None):
        self.opts = opts or AskOptions()
        self.store = store or open_store(index)
        self.llm: LLM = get_llm(llm) if (llm is None or isinstance(llm, str)) else llm
        reranker = CrossEncoderReranker(self.opts.reranker_model) if self.opts.rerank else None
        self.retriever = Retriever(self.store, mode=self.opts.mode, reranker=reranker,
                                   rewriter=self.llm if self.opts.rewrite else None)
        self.trace_path = trace_path

    @property
    def web_base(self) -> str | None:
        return self.store.meta.get("web_base")

    def retrieve(self, question: str, history: list[dict] | None = None) -> tuple[str, list[Chunk]]:
        standalone = condense(history, question, self.llm) if history else question
        hits = self.retriever.search(standalone, self.opts.k)
        sources = pack_context(hits, self.opts.budget, self.store, expand=self.opts.expand)
        return standalone, sources

    def stream(self, question: str, history: list[dict] | None = None) -> Iterator[dict]:
        """Yields {"type": "sources"|"token"|"done", ...} events."""
        t0 = time.perf_counter()
        standalone, sources = self.retrieve(question, history)
        yield {"type": "sources", "standalone_question": standalone,
               "sources": [self._source_json(i, c) for i, c in enumerate(sources, 1)]}
        parts = []
        for piece in stream_answer(standalone, sources, self.llm, history):
            parts.append(piece)
            yield {"type": "token", "text": piece}
        ans = finalize(question, "".join(parts), sources, self.llm.name, self.web_base, standalone)
        self._log(ans, time.perf_counter() - t0)
        yield {"type": "done", **self.answer_json(ans)}

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        ans = None
        for ev in self.stream(question, history):
            if ev["type"] == "done":
                ans = ev["_answer"]
        return ans

    def _source_json(self, i: int, c: Chunk) -> dict:
        from .loader import permalink

        return {"id": f"S{i}", "path": c.path, "start_line": c.start_line, "end_line": c.end_line,
                "symbol": c.symbol, "text": c.text, "url": permalink(self.web_base, c.path, c.start_line, c.end_line)}

    def answer_json(self, ans: Answer) -> dict:
        text, notes = ans.rendered()
        return {
            "_answer": ans,
            "answer": text,
            "raw": ans.text,
            "footnotes": notes,
            "abstained": ans.abstained,
            "invalid_citations": [asdict(c) for c in ans.invalid_citations],
            "unsupported_identifiers": ans.unsupported_identifiers,
            "uncited": not ans.citations and not ans.abstained,
            "model": ans.model,
        }

    def _log(self, ans: Answer, seconds: float) -> None:
        if not self.trace_path:
            return
        rec = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "question": ans.question,
            "standalone_question": ans.standalone_question,
            "retrieval": self.retriever.last_trace,
            "sources": [c.citation for c in ans.sources],
            "answer": ans.text,
            "citations": [asdict(c) for c in ans.citations],
            "unsupported_identifiers": ans.unsupported_identifiers,
            "abstained": ans.abstained,
            "seconds": round(seconds, 2),
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.trace_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
