"""Query-side LLM helpers: query rewriting (keywords + HyDE) and follow-up condensation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import LLM

REWRITE_SYSTEM = (
    "You turn a developer's question about a code repository into search inputs. Return JSON with:\n"
    '  "keywords": 3-8 likely code identifiers or exact terms (method, class, setting or constant names, '
    "in their likely casing) that the answer's source code would contain;\n"
    '  "hypothetical_code": a short (5-15 line) snippet of what the relevant code plausibly looks like.\n'
    "Guess plausible names; it is fine to be wrong. Output JSON only."
)

CONDENSE_SYSTEM = (
    "Rewrite the user's latest question so it can be understood without the conversation, resolving "
    "pronouns and references (\"it\", \"that method\") using the earlier turns. Keep code identifiers "
    "exactly. If it is already self-contained, return it unchanged. Output only the rewritten question."
)


@dataclass
class QueryPlan:
    question: str
    keywords: list[str] = field(default_factory=list)
    hypothetical_code: str = ""

    @property
    def keyword_query(self) -> str:
        return " ".join([self.question, *self.keywords])


def plan_query(question: str, llm: LLM) -> QueryPlan:
    try:
        data = json.loads(llm.complete(REWRITE_SYSTEM, question, json_mode=True))
    except (json.JSONDecodeError, RuntimeError):
        return QueryPlan(question)
    if not isinstance(data, dict):
        return QueryPlan(question)
    kws = [str(k) for k in data.get("keywords", []) if isinstance(k, (str, int))][:8]
    return QueryPlan(question, kws, str(data.get("hypothetical_code", ""))[:2000])


def condense(history: list[dict], question: str, llm: LLM, max_turns: int = 6) -> str:
    """history: [{"role": "user"|"assistant", "content": str}, ...]"""
    if not history:
        return question
    convo = "\n".join(f"{t['role'].upper()}: {t['content'][:1500]}" for t in history[-max_turns:])
    out = llm.complete(CONDENSE_SYSTEM, f"Conversation:\n{convo}\n\nLatest question: {question}").strip()
    return out.splitlines()[0].strip() if out else question
