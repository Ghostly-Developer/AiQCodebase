"""LLM-as-judge: grade an answer against a reference answer.

Keyword matching is cheap but brittle (a wrong answer can contain "100" by accident, a right answer
can paraphrase). A judge with a reference answer is closer to how a person would grade.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .llm import LLM

JUDGE_SYSTEM = """You grade answers from a code Q&A assistant against a reference answer written by an expert.

Verdicts:
- "correct": states the key facts of the reference; extra correct detail is fine.
- "partial": gets some key facts right but misses or garbles others.
- "wrong": contradicts the reference, misses the point, or invents behaviour.
- "abstained": the answer says it doesn't know / can't find it.

For questions marked UNANSWERABLE the repository does not contain the feature: an answer that
abstains or clearly says the feature doesn't exist is "correct"; any answer describing how to do it is "wrong".

Ignore citation markers like [1] or [S2:L10-L12]. Judge substance, not style.
Return JSON: {"verdict": "correct"|"partial"|"wrong"|"abstained", "reason": "<one sentence>"}"""

SCORES = {"correct": 1.0, "partial": 0.5, "wrong": 0.0, "abstained": 0.0}


@dataclass
class Verdict:
    verdict: str
    reason: str

    @property
    def score(self) -> float:
        return SCORES.get(self.verdict, 0.0)


def judge(question: str, reference: str | None, answer: str, llm: LLM, answerable: bool = True) -> Verdict:
    ref = reference if answerable else "UNANSWERABLE: the repository has no such feature; the right response is to say so."
    prompt = f"Question: {question}\n\nReference answer: {ref}\n\nAnswer to grade:\n{answer}"
    try:
        data = json.loads(llm.complete(JUDGE_SYSTEM, prompt, json_mode=True))
        v = str(data.get("verdict", "wrong")).lower()
        return Verdict(v if v in SCORES else "wrong", str(data.get("reason", "")))
    except (json.JSONDecodeError, RuntimeError, AttributeError) as e:
        return Verdict("wrong", f"judge error: {e}")
