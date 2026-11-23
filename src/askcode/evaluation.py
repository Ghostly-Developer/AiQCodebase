"""Evaluation: score retrieval and answers against questions with known answers.

An eval set is a JSON file:

    {
      "repo": "https://github.com/DapperLib/Dapper.git",   # or a local path
      "ref": "<commit sha>",                                 # pin it so line numbers never drift
      "exclude": ["tests/*"],
      "questions": [
        {"id": "q01", "question": "...", "category": "locate",
         "expected": [{"path": "Dapper/SqlMapper.cs", "contains": "public static int Execute("}],
         "answer_keywords": ["ExecuteImpl"],
         "reference": "One or two sentences an expert would accept as correct."},
        {"id": "u01", "question": "How do I configure the Redis cache?", "answerable": false}
      ]
    }

``expected`` holds evidence locations: a file plus a literal snippet, resolved to a line number at eval
time so the set stays readable. A retrieved chunk "hits" when it is in that file and its line range
contains the evidence line. Unanswerable questions (``"answerable": false``) test whether the
assistant says "I don't know" instead of inventing an answer.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .answer import answer_question
from .judge import judge
from .models import Chunk, Hit, approx_tokens
from .retrieval import Retriever, pack_context

CITE_TOLERANCE = 5  # a citation within +-5 lines of the evidence line counts as pointing at it


@dataclass
class Evidence:
    path: str
    contains: str
    line: int = 0


@dataclass
class EvalItem:
    id: str
    question: str
    expected: list[Evidence]
    answer_keywords: list[str]
    category: str = ""
    reference: str | None = None
    answerable: bool = True


@dataclass
class EvalSet:
    repo: str
    ref: str | None
    include: list[str] | None
    exclude: list[str] | None
    items: list[EvalItem]
    path: Path

    @property
    def name(self) -> str:
        return self.path.stem

    @property
    def answerable(self) -> list[EvalItem]:
        return [i for i in self.items if i.answerable]


def load_eval(path: str | Path) -> EvalSet:
    path = Path(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    items = [
        EvalItem(
            id=q["id"],
            question=q["question"],
            expected=[Evidence(e["path"], e["contains"]) for e in q.get("expected", [])],
            answer_keywords=q.get("answer_keywords", []),
            category=q.get("category", "" if q.get("answerable", True) else "unanswerable"),
            reference=q.get("reference"),
            answerable=q.get("answerable", True),
        )
        for q in d["questions"]
    ]
    return EvalSet(d["repo"], d.get("ref"), d.get("include"), d.get("exclude"), items, path)


def resolve_evidence(es: EvalSet, root: Path) -> list[str]:
    """Fill in Evidence.line from the snippet. Returns a list of problems (empty = all good)."""
    problems = []
    cache: dict[str, list[str]] = {}
    for item in es.items:
        if item.answerable and not item.expected:
            problems.append(f"{item.id}: answerable question has no expected evidence")
        for ev in item.expected:
            if ev.path not in cache:
                p = root / ev.path
                cache[ev.path] = p.read_text(encoding="utf-8-sig").replace("\r\n", "\n").split("\n") if p.exists() else []
            lines = cache[ev.path]
            if not lines:
                problems.append(f"{item.id}: file not found: {ev.path}")
                continue
            matches = [i for i, l in enumerate(lines, 1) if ev.contains in l]
            if not matches:
                problems.append(f"{item.id}: snippet not found in {ev.path}: {ev.contains!r}")
            else:
                ev.line = matches[0]
                if len(matches) > 1:
                    problems.append(f"{item.id}: warning: snippet matches {len(matches)} lines in {ev.path}, using first")
    return problems


def _hits_evidence(c: Chunk, item: EvalItem) -> bool:
    return any(c.path == e.path and c.start_line <= e.line <= c.end_line for e in item.expected)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("`", "")).lower()


def keywords_present(text: str, keywords: list[str]) -> bool:
    """Every keyword must appear (case-insensitive). 'a|b' means either alternative is fine."""
    t = _normalize(text)
    return all(any(_normalize(alt) in t for alt in k.split("|")) for k in keywords)


@dataclass
class QuestionResult:
    id: str
    category: str
    answerable: bool
    rank: int | None  # 1-based rank of first chunk containing evidence (None = not in top-k)
    file_rank: int | None
    in_context: bool  # evidence line survived packing into the token budget
    context_tokens: int
    top: list[str]
    retriever_ranks: dict[str, int | None] = field(default_factory=dict)  # evidence rank in each retriever's list
    answer: str | None = None
    keyword_correct: bool | None = None
    judge_verdict: str | None = None
    judge_reason: str | None = None
    judge_score: float | None = None
    cited_evidence: bool | None = None
    abstained: bool | None = None
    invalid_citations: int | None = None
    unsupported_identifiers: list[str] = field(default_factory=list)


@dataclass
class RunSummary:
    label: str
    n: int  # answerable questions (retrieval metrics are over these)
    n_unanswerable: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    file_at_5: float
    mrr_at_10: float
    hit_at_budget: float
    avg_context_tokens: float
    avg_chunk_tokens: float
    chunks: int
    ci: dict[str, list[float]] = field(default_factory=dict)  # metric -> [low, high] (95% bootstrap)
    keyword_accuracy: float | None = None
    judge_score: float | None = None
    cited_accuracy: float | None = None
    false_refusal: float | None = None
    refusal_accuracy: float | None = None  # unanswerable questions correctly declined
    invalid_citations_per_answer: float | None = None
    unsupported_ids_per_answer: float | None = None
    seconds: float = 0.0


def bootstrap_ci(values: list[float], iters: int = 2000, seed: int = 0) -> list[float]:
    """95% percentile bootstrap interval for the mean."""
    if not values:
        return [0.0, 0.0]
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(iters, len(a)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def run_eval(
    es: EvalSet,
    retriever: Retriever,
    label: str,
    k: int = 10,
    budget_tokens: int = 2000,
    llm=None,
    judge_llm=None,
    expand: bool = False,
    progress=None,
) -> tuple[RunSummary, list[QuestionResult]]:
    t0 = time.time()
    results: list[QuestionResult] = []
    items = es.items if llm is not None else es.answerable  # unanswerable only matter for answers
    for n, item in enumerate(items, 1):
        hits: list[Hit] = retriever.search(item.question, k=k)
        packed = pack_context(hits, budget_tokens, retriever.store, expand=expand)
        r = QuestionResult(
            id=item.id, category=item.category, answerable=item.answerable, rank=None, file_rank=None,
            in_context=False, context_tokens=sum(approx_tokens(c.text) for c in packed),
            top=[h.chunk.citation for h in hits[:5]],
        )
        if item.answerable:
            r.rank = next((i for i, h in enumerate(hits, 1) if _hits_evidence(h.chunk, item)), None)
            paths = {e.path for e in item.expected}
            r.file_rank = next((i for i, h in enumerate(hits, 1) if h.chunk.path in paths), None)
            r.in_context = any(_hits_evidence(c, item) for c in packed)
            r.retriever_ranks = {
                src: next((i for i, (c, _) in enumerate(lst, 1) if _hits_evidence(c, item)), None)
                for src, lst in retriever.last_lists.items()
            }
        if llm is not None:
            ans = answer_question(item.question, packed, llm, web_base=retriever.store.meta.get("web_base"))
            r.answer = ans.text
            r.abstained = ans.abstained
            r.invalid_citations = len(ans.invalid_citations)
            r.unsupported_identifiers = ans.unsupported_identifiers
            if item.answerable:
                r.keyword_correct = keywords_present(ans.text, item.answer_keywords) and not ans.abstained
                r.cited_evidence = any(
                    c.path == e.path and c.start_line - CITE_TOLERANCE <= e.line <= c.end_line + CITE_TOLERANCE
                    for c in ans.valid_citations for e in item.expected
                )
            if judge_llm is not None:
                v = judge(item.question, item.reference, ans.text, judge_llm, item.answerable)
                r.judge_verdict, r.judge_reason, r.judge_score = v.verdict, v.reason, v.score
        results.append(r)
        if progress:
            progress(n, len(items))

    ans_results = [r for r in results if r.answerable]
    una_results = [r for r in results if not r.answerable]

    def vals(pred, rs=None) -> list[float]:
        return [1.0 if pred(r) else 0.0 for r in (ans_results if rs is None else rs)]

    def mean(v: list[float]) -> float:
        return sum(v) / len(v) if v else 0.0

    r5 = vals(lambda r: r.rank is not None and r.rank <= 5)
    hit = vals(lambda r: r.in_context)
    chunks = retriever.store.chunks()
    summary = RunSummary(
        label=label,
        n=len(ans_results),
        n_unanswerable=len(una_results),
        recall_at_1=mean(vals(lambda r: r.rank == 1)),
        recall_at_5=mean(r5),
        recall_at_10=mean(vals(lambda r: r.rank is not None and r.rank <= 10)),
        file_at_5=mean(vals(lambda r: r.file_rank is not None and r.file_rank <= 5)),
        mrr_at_10=mean([1 / r.rank if r.rank else 0.0 for r in ans_results]),
        hit_at_budget=mean(hit),
        avg_context_tokens=mean([r.context_tokens for r in ans_results]),
        avg_chunk_tokens=sum(approx_tokens(c.text) for c in chunks) / max(1, len(chunks)),
        chunks=len(chunks),
        ci={"recall_at_5": bootstrap_ci(r5), "hit_at_budget": bootstrap_ci(hit)},
    )
    if llm is not None:
        kw = vals(lambda r: bool(r.keyword_correct))
        summary.keyword_accuracy = mean(kw)
        summary.cited_accuracy = mean(vals(lambda r: bool(r.keyword_correct) and bool(r.cited_evidence)))
        summary.false_refusal = mean(vals(lambda r: bool(r.abstained)))
        summary.invalid_citations_per_answer = mean([r.invalid_citations or 0 for r in results])
        summary.unsupported_ids_per_answer = mean([len(r.unsupported_identifiers) for r in results])
        if una_results:
            if judge_llm is not None:
                refused = [1.0 if r.judge_verdict == "correct" else 0.0 for r in una_results]
            else:
                refused = vals(lambda r: bool(r.abstained), una_results)
            summary.refusal_accuracy = mean(refused)
            summary.ci["refusal_accuracy"] = bootstrap_ci(refused)
        if judge_llm is not None:
            js = [r.judge_score or 0.0 for r in ans_results]
            summary.judge_score = mean(js)
            summary.ci["judge_score"] = bootstrap_ci(js)
    summary.seconds = time.time() - t0
    return summary, results


# --------------------------------------------------------------------------- reporting

def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.0f}%"


def _pct_ci(s: RunSummary, metric: str) -> str:
    v = getattr(s, metric)
    lo, hi = s.ci.get(metric, [None, None])
    if v is None:
        return "-"
    return f"{100 * v:.0f}% ({100 * lo:.0f}–{100 * hi:.0f})" if lo is not None else _pct(v)


def markdown_table(summaries: list[RunSummary], budget_tokens: int) -> str:
    head = ["Config", "Chunks", "Avg chunk tok", "Recall@1", "Recall@5 (95% CI)", "File@5", "MRR@10",
            f"Hit@{budget_tokens}tok (95% CI)"]
    rows = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for s in summaries:
        rows.append("| " + " | ".join([
            s.label, str(s.chunks), f"{s.avg_chunk_tokens:.0f}", _pct(s.recall_at_1), _pct_ci(s, "recall_at_5"),
            _pct(s.file_at_5), f"{s.mrr_at_10:.2f}", _pct_ci(s, "hit_at_budget"),
        ]) + " |")
    return "\n".join(rows)


def answers_table(summaries: list[RunSummary]) -> str:
    head = ["Config", "Judge score (95% CI)", "Keyword acc", "Correct + cited", "Refused unanswerable",
            "False refusals", "Bad citations / answer", "Unsupported ids / answer"]
    rows = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for s in summaries:
        rows.append("| " + " | ".join([
            s.label, _pct_ci(s, "judge_score"), _pct(s.keyword_accuracy), _pct(s.cited_accuracy),
            _pct_ci(s, "refusal_accuracy"), _pct(s.false_refusal),
            "-" if s.invalid_citations_per_answer is None else f"{s.invalid_citations_per_answer:.2f}",
            "-" if s.unsupported_ids_per_answer is None else f"{s.unsupported_ids_per_answer:.2f}",
        ]) + " |")
    return "\n".join(rows)


def load_results(path: Path) -> tuple[dict, list[dict]]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return d["summary"], d["results"]


METRICS = {
    "hit": lambda r: 1.0 if r["in_context"] else 0.0,
    "recall5": lambda r: 1.0 if r["rank"] is not None and r["rank"] <= 5 else 0.0,
    "mrr": lambda r: 1.0 / r["rank"] if r["rank"] else 0.0,
    "judge": lambda r: r.get("judge_score") or 0.0,
}


def sign_test_p(wins: int, losses: int) -> float:
    """Exact two-sided sign test: probability of a split at least this lopsided if A and B were equal."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2**n
    return min(1.0, 2 * p)


def paired_compare(a: list[dict], b: list[dict], metric: str = "hit") -> dict:
    """Per-question comparison of two runs on the same questions (B relative to A)."""
    f = METRICS[metric]
    a_by = {r["id"]: r for r in a if r.get("answerable", True)}
    b_by = {r["id"]: r for r in b if r.get("answerable", True)}
    common = sorted(set(a_by) & set(b_by))
    diffs = [f(b_by[i]) - f(a_by[i]) for i in common]
    wins = [i for i, d in zip(common, diffs) if d > 0]
    losses = [i for i, d in zip(common, diffs) if d < 0]
    lo, hi = bootstrap_ci(diffs)
    return {
        "n": len(common), "metric": metric, "mean_a": float(np.mean([f(a_by[i]) for i in common])),
        "mean_b": float(np.mean([f(b_by[i]) for i in common])), "mean_diff": float(np.mean(diffs)) if diffs else 0.0,
        "diff_ci": [lo, hi], "b_better": wins, "b_worse": losses, "p_sign_test": sign_test_p(len(wins), len(losses)),
    }


def category_report(results: list[dict]) -> str:
    cats: dict[str, list[dict]] = {}
    for r in results:
        cats.setdefault(r.get("category") or "-", []).append(r)
    has_judge = any(r.get("judge_score") is not None for r in results)
    head = ["Category", "n", "Recall@5", "Hit@budget"] + (["Judge"] if has_judge else [])
    rows = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for cat, rs in sorted(cats.items()):
        ans = [r for r in rs if r.get("answerable", True)]
        row = [cat, str(len(rs)),
               _pct(np.mean([METRICS["recall5"](r) for r in ans])) if ans else "-",
               _pct(np.mean([METRICS["hit"](r) for r in ans])) if ans else "-"]
        if has_judge:
            js = [r.get("judge_score") for r in rs if r.get("judge_score") is not None]
            row.append(_pct(np.mean(js)) if js else "-")
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def failure_report(results: list[dict]) -> list[dict]:
    """Answerable questions whose evidence didn't make it into the context, with where each retriever ranked it."""
    out = []
    for r in results:
        if r.get("answerable", True) and not r["in_context"]:
            out.append({"id": r["id"], "category": r.get("category"), "rank": r["rank"],
                        "retriever_ranks": r.get("retriever_ranks", {}), "top": r["top"][:3]})
    return out


def update_readme(readme: Path, marker: str, content: str) -> bool:
    """Replace the text between <!-- {marker}:START --> and <!-- {marker}:END --> in README."""
    text = readme.read_text(encoding="utf-8")
    start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pattern.search(text):
        return False
    readme.write_text(pattern.sub(lambda _: f"{start}\n{content}\n{end}", text), encoding="utf-8")
    return True
