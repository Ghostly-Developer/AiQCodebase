"""Web UI: `askcode serve`. FastAPI + a single static page; answers stream over Server-Sent Events."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .pipeline import AskOptions, Assistant
from .store import list_local

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Ask My Codebase")
_assistants: dict[tuple, Assistant] = {}
_lock = threading.Lock()
TRACE_PATH: Path | None = None


class Turn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    index: str
    question: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = []
    mode: str = "hybrid"
    rerank: bool = False
    rewrite: bool = False
    expand: bool = True
    llm: str | None = None
    k: int = Field(8, ge=1, le=30)
    budget: int = Field(3000, ge=500, le=30000)


def _assistant(req: AskRequest) -> Assistant:
    key = (req.index, req.llm, req.mode, req.rerank, req.rewrite, req.expand, req.k, req.budget)
    with _lock:
        if key not in _assistants:
            opts = AskOptions(mode=req.mode, rerank=req.rerank, rewrite=req.rewrite, expand=req.expand,
                              k=req.k, budget=req.budget)
            _assistants[key] = Assistant(req.index, req.llm, opts, trace_path=TRACE_PATH)
        return _assistants[key]


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/indexes")
def indexes() -> list[dict]:
    return [{"name": m["name"], "source": m["source"], "chunker": m["chunker"]["name"], "chunks": m["chunks"],
             "commit": (m.get("commit") or "")[:10], "web_base": m.get("web_base")} for m in list_local()]


@app.post("/api/ask")
def ask(req: AskRequest) -> StreamingResponse:
    try:
        assistant = _assistant(req)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e)) from e
    history = [t.model_dump() for t in req.history]

    def events():
        try:
            for ev in assistant.stream(req.question, history):
                ev.pop("_answer", None)
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:  # surface provider errors (rate limits, bad key) to the UI
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)[:500]})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def run(host: str = "127.0.0.1", port: int = 8000, trace: Path | None = None) -> None:
    import uvicorn

    global TRACE_PATH
    TRACE_PATH = trace
    uvicorn.run(app, host=host, port=port, log_level="warning")
