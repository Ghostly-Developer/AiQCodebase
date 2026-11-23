from __future__ import annotations

import json
import random
import re
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from . import config
from .indexer import build_index, update_index
from .retrieval import DEFAULT_RERANKER, CrossEncoderReranker, Retriever, pack_context
from .store import list_local, open_store

app = typer.Typer(help="Ask My Codebase: RAG over a repo with file:line citations.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluate retrieval and answers against a question set.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")
console = Console()


def _llm_or_exit(spec: str | None):
    from .llm import get_llm

    try:
        return get_llm(spec)
    except RuntimeError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e


# --------------------------------------------------------------------------- indexing


@app.command()
def index(
    source: str = typer.Argument(..., help="Local path or git URL"),
    name: str = typer.Option(..., "--name", "-n", help="Index name"),
    chunker: str = typer.Option("ast", help="ast | function | fixed | file"),
    embedder: str = typer.Option("fastembed", help="fastembed[:model] | google[:model] | hash"),
    store: str = typer.Option("local", help="local | pgvector"),
    ref: str = typer.Option(None, help="Branch, tag or commit to check out (git URLs only)"),
    include: list[str] = typer.Option(None, help="Glob of paths to include (repeatable)"),
    exclude: list[str] = typer.Option(None, help="Glob of paths to exclude (repeatable)"),
    contextualize: bool = typer.Option(False, help="Have an LLM summarise each chunk before embedding"),
    llm: str = typer.Option(None, help="LLM for --contextualize: google | groq | <provider>:<model>"),
):
    """Chunk, embed and store a repository."""
    ctx_llm = _llm_or_exit(llm) if contextualize else None
    _, stats = build_index(name, source, chunker, embedder, store, ref, include, exclude,
                           contextualize_llm=ctx_llm, progress=console.print)
    console.print(
        f"[green]Indexed[/] {stats.files} files -> {stats.chunks} chunks "
        f"(avg {stats.avg_tokens:.0f} tok, max {stats.max_tokens}) in {stats.seconds:.1f}s as [bold]{name}[/]"
        + (f", {stats.redactions} secret(s) redacted" if stats.redactions else "")
    )


@app.command()
def update(
    name: str = typer.Argument(..., help="Index to refresh"),
    llm: str = typer.Option(None, help="LLM for chunk summaries, if the index was built with --contextualize"),
):
    """Re-index only the files that changed since the index was built (content-hash based)."""
    meta = open_store(name).meta
    ctx_llm = _llm_or_exit(llm) if meta.get("contextualized") else None
    _, stats = update_index(name, contextualize_llm=ctx_llm, progress=console.print)
    if not stats.changed and not stats.removed:
        console.print(f"[green]Up to date[/]: {stats.files} files, {stats.chunks} chunks.")
        return
    console.print(f"[green]Updated[/] {len(stats.changed)} changed / {len(stats.removed)} removed file(s) "
                  f"in {stats.seconds:.1f}s; index now has {stats.chunks} chunks.")
    for p in (stats.changed + stats.removed)[:20]:
        console.print(f"  {p}")


@app.command("list")
def list_indexes():
    """List local indexes."""
    t = Table("Name", "Chunker", "Embedder", "Files", "Chunks", "Commit", "Source")
    for m in list_local():
        t.add_row(m["name"], m["chunker"]["name"] + (" +ctx" if m.get("contextualized") else ""), m["embedder"],
                  str(m["files"]), str(m["chunks"]), (m.get("commit") or "")[:10], m["source"])
    console.print(t)


# --------------------------------------------------------------------------- querying


@app.command()
def search(
    query: str,
    index: str = typer.Option(..., "--index", "-i"),
    mode: str = typer.Option("hybrid", help="vector | keyword | hybrid"),
    rerank: bool = typer.Option(False, help="Rerank candidates with a cross-encoder"),
    reranker_model: str = typer.Option(DEFAULT_RERANKER),
    rewrite: bool = typer.Option(False, help="LLM query rewriting (keywords + hypothetical code)"),
    llm: str = typer.Option(None),
    k: int = typer.Option(8),
    trace: bool = typer.Option(False, help="Print the full retrieval trace as JSON"),
):
    """Retrieve chunks only (no answer generation; no API key needed unless --rewrite)."""
    store = open_store(index)
    r = Retriever(store, mode=mode, reranker=CrossEncoderReranker(reranker_model) if rerank else None,
                  rewriter=_llm_or_exit(llm) if rewrite else None)
    t = Table("#", "Score", "Location", "Symbol", "Found by")
    for i, h in enumerate(r.search(query, k), 1):
        t.add_row(str(i), f"{h.score:.3f}", h.chunk.citation, h.chunk.symbol or "",
                  ", ".join(f"{k}#{v}" for k, v in h.ranks.items()))
    console.print(t)
    if trace:
        console.print_json(json.dumps(r.last_trace))


def _print_answer(ev: dict, show_sources: bool, sources: list[dict]) -> None:
    if show_sources:
        for s in sources:
            console.print(Panel(s["text"], title=f"{s['id']} {s['path']}:L{s['start_line']}-L{s['end_line']} {s['symbol'] or ''}",
                                expand=False))
    console.print(Markdown(ev["answer"]))
    if ev["footnotes"]:
        console.print("\n[bold]Sources[/]")
        for n in ev["footnotes"]:
            flag = "" if n["valid"] else " [yellow](unverified)[/]"
            link = f"  [link={n['url']}]{n['url']}[/link]" if n.get("url") else ""
            console.print(f"  [{n['n']}] {n['ref']}{flag}{link}")
    if ev["invalid_citations"]:
        console.print(f"[yellow]{len(ev['invalid_citations'])} citation(s) don't match the sources:[/] "
                      + "; ".join(c["reason"] for c in ev["invalid_citations"]))
    if ev["unsupported_identifiers"]:
        console.print("[yellow]Identifiers not found in any source (possible hallucination):[/] "
                      + ", ".join(ev["unsupported_identifiers"]))
    if ev["uncited"]:
        console.print("[yellow]The answer has no citations, so treat it as unverified.[/]")
    console.print(f"[dim]{ev['model']} - {len(sources)} sources[/]")


def _assistant(index, llm, mode, rerank, reranker_model, rewrite, expand, k, budget, trace):
    from .pipeline import AskOptions, Assistant

    opts = AskOptions(mode=mode, rerank=rerank, reranker_model=reranker_model, rewrite=rewrite, expand=expand,
                      k=k, budget=budget)
    return Assistant(index, _llm_or_exit(llm), opts, trace_path=trace)


@app.command()
def ask(
    question: str,
    index: str = typer.Option(..., "--index", "-i"),
    mode: str = typer.Option("hybrid", help="vector | keyword | hybrid"),
    rerank: bool = typer.Option(False, help="Rerank candidates with a cross-encoder"),
    reranker_model: str = typer.Option(DEFAULT_RERANKER),
    rewrite: bool = typer.Option(False, help="LLM query rewriting (keywords + hypothetical code)"),
    expand: bool = typer.Option(True, help="Small-to-big: give the LLM whole methods / neighbouring windows"),
    k: int = typer.Option(8, help="Chunks to retrieve"),
    budget: int = typer.Option(3000, help="Context token budget"),
    llm: str = typer.Option(None, help="google | groq | google:<model> | groq:<model>"),
    show_sources: bool = typer.Option(False, help="Print the retrieved context"),
    trace: Path = typer.Option(None, help="Append a JSONL trace of this query to this file"),
):
    """Answer a question with citations to file and line."""
    a = _assistant(index, llm, mode, rerank, reranker_model, rewrite, expand, k, budget, trace)
    sources: list[dict] = []
    with console.status("Thinking..."):
        for ev in a.stream(question):
            if ev["type"] == "sources":
                sources = ev["sources"]
            elif ev["type"] == "done":
                done = ev
    _print_answer(done, show_sources, sources)


@app.command()
def chat(
    index: str = typer.Option(..., "--index", "-i"),
    mode: str = typer.Option("hybrid"),
    rerank: bool = typer.Option(False),
    rewrite: bool = typer.Option(False),
    expand: bool = typer.Option(True),
    k: int = typer.Option(8),
    budget: int = typer.Option(3000),
    llm: str = typer.Option(None),
    trace: Path = typer.Option(None),
):
    """Interactive Q&A with follow-up questions. Empty line or Ctrl+C to quit."""
    a = _assistant(index, llm, mode, rerank, DEFAULT_RERANKER, rewrite, expand, k, budget, trace)
    history: list[dict] = []
    console.print(f"[bold]Chatting with {index}[/] ({a.llm.name}). Follow-ups like 'and the async version?' work.")
    while True:
        try:
            q = console.input("\n[bold cyan]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        sources: list[dict] = []
        with console.status("Thinking..."):
            for ev in a.stream(q, history):
                if ev["type"] == "sources":
                    sources = ev["sources"]
                    if ev["standalone_question"] != q:
                        console.print(f"[dim]searching for: {ev['standalone_question']}[/]")
                elif ev["type"] == "done":
                    done = ev
        _print_answer(done, False, sources)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": done["raw"]}]


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    trace: Path = typer.Option(None, help="Append a JSONL trace of every query to this file"),
):
    """Start the web UI (streamed answers, follow-ups, clickable citations)."""
    from .server import run

    console.print(f"Open http://{host}:{port}")
    run(host, port, trace)


# --------------------------------------------------------------------------- evaluation


@eval_app.command("check")
def eval_check(eval_file: Path):
    """Validate an eval set: every evidence snippet must resolve to a line in the pinned repo."""
    from . import loader
    from .evaluation import load_eval, resolve_evidence

    es = load_eval(eval_file)
    repo = loader.load(es.repo, ref=es.ref, include=es.include, exclude=es.exclude)
    problems = resolve_evidence(es, repo.root)
    for p in problems:
        console.print(("[yellow]" if "warning" in p else "[red]") + p)
    errors = [p for p in problems if "warning" not in p]
    missing_ref = [i.id for i in es.answerable if not i.reference]
    if missing_ref:
        console.print(f"[yellow]{len(missing_ref)} answerable question(s) have no reference answer (needed for the judge)[/]")
    console.print(f"{len(es.answerable)} answerable + {len(es.items) - len(es.answerable)} unanswerable questions, "
                  f"{len(errors)} errors, {len(problems) - len(errors)} warnings")
    raise typer.Exit(1 if errors else 0)


def _parse_mode(token: str) -> dict:
    """'hybrid+rerank:<model>+rewrite+expand' -> {'base': 'hybrid', 'rerank': model|None, 'rewrite': bool, 'expand': bool}"""
    base, *flags = token.split("+")
    out = {"base": base, "rerank": None, "rewrite": False, "expand": False}
    for f in flags:
        name, _, arg = f.partition(":")
        if name == "rerank":
            out["rerank"] = arg or DEFAULT_RERANKER
        elif name in ("rewrite", "expand"):
            out[name] = True
        else:
            raise typer.BadParameter(f"unknown mode flag {f!r} (use rerank[:model], rewrite, expand)")
    return out


@eval_app.command("run")
def eval_run(
    eval_file: Path,
    chunkers: str = typer.Option("file,fixed,function,ast", help="Comma-separated chunkers to compare"),
    modes: str = typer.Option("hybrid", help="Comma-separated, e.g. vector,keyword,hybrid,hybrid+rerank,hybrid+expand,hybrid+rewrite"),
    embedder: str = typer.Option("fastembed"),
    contextualize: bool = typer.Option(False, help="Use indexes whose chunks were summarised by an LLM"),
    k: int = typer.Option(10),
    budget: int = typer.Option(2000, help="Context token budget for Hit@budget and answers"),
    answers: bool = typer.Option(False, help="Also generate and score answers (needs an LLM API key)"),
    llm: str = typer.Option(None, help="Answer LLM: google | groq | <provider>:<model>"),
    judge_llm: str = typer.Option(None, help="Judge LLM (default: same as --llm). Use 'none' for keyword scoring only"),
    limit: int = typer.Option(0, help="Only run the first N questions (0 = all)"),
    reindex: bool = typer.Option(False, help="Rebuild indexes even if they exist"),
    out: Path = typer.Option(Path("evals/results"), help="Where to write per-question JSON"),
    readme: Path = typer.Option(None, help="README to update between <!-- MARKER:START/END --> markers"),
    marker: str = typer.Option("EVAL", help="Marker name in the README for the retrieval table"),
    answers_marker: str = typer.Option("ANSWERS", help="Marker name for the answer-quality table"),
):
    """Build one index per chunker, then score every (chunker, mode) combination."""
    from .embeddings import get_embedder
    from .evaluation import answers_table, load_eval, markdown_table, resolve_evidence, run_eval, update_readme

    es = load_eval(eval_file)
    if limit:
        es.items = es.items[:limit]
    model = _llm_or_exit(llm) if (answers or contextualize or "rewrite" in modes) else None
    judge = None
    if answers and judge_llm != "none":
        judge = _llm_or_exit(judge_llm) if judge_llm else model
    emb = get_embedder(embedder)
    slug = re.sub(r"[^a-z0-9]+", "-", embedder.lower()).strip("-")
    emb_label = "" if embedder == "fastembed" else f" [{embedder.split('/')[-1]}]"
    summaries = []
    rerankers: dict[str, CrossEncoderReranker] = {}
    out.mkdir(parents=True, exist_ok=True)

    for chunker in [c.strip() for c in chunkers.split(",") if c.strip()]:
        name = f"{es.name}-{chunker}-{slug}" + ("-ctx" if contextualize else "")
        try:
            if reindex:
                raise FileNotFoundError
            store = open_store(name, "local")
        except FileNotFoundError:
            console.print(f"Building index [bold]{name}[/] ...")
            store, stats = build_index(name, es.repo, chunker, embedder, "local", es.ref, es.include, es.exclude,
                                       contextualize_llm=model if contextualize else None)
            console.print(f"  {stats.files} files -> {stats.chunks} chunks in {stats.seconds:.0f}s")
        problems = [p for p in resolve_evidence(es, Path(store.meta["root"])) if "warning" not in p]
        if problems:
            console.print("[red]Eval set has unresolved evidence:[/]\n" + "\n".join(problems))
            raise typer.Exit(1)

        for mode in [m.strip() for m in modes.split(",") if m.strip()]:
            m = _parse_mode(mode)
            reranker = None
            if m["rerank"]:
                reranker = rerankers.setdefault(m["rerank"], CrossEncoderReranker(m["rerank"]))
            r = Retriever(store, mode=m["base"], reranker=reranker, embedder=emb,
                          rewriter=model if m["rewrite"] else None)
            label = f"{chunker}{' +ctx' if contextualize else ''} / {mode}{emb_label}"
            with Progress(console=console, transient=True) as prog:
                task = prog.add_task(label, total=len(es.items))
                summary, results = run_eval(es, r, label, k, budget, model if answers else None, judge,
                                            expand=m["expand"], progress=lambda n, t: prog.update(task, completed=n, total=t))
            summaries.append(summary)
            safe_mode = re.sub(r"[^a-zA-Z0-9]+", "-", mode)
            fname = out / f"{es.name}__{chunker}{'-ctx' if contextualize else ''}__{safe_mode}__{slug}{'__answers' if answers else ''}.json"
            fname.write_text(json.dumps({"summary": asdict(summary), "results": [asdict(x) for x in results]},
                                        indent=2), encoding="utf-8")
            line = (f"  {label}: Recall@5 {summary.recall_at_5:.0%}, MRR {summary.mrr_at_10:.2f}, "
                    f"Hit@budget {summary.hit_at_budget:.0%}")
            if summary.judge_score is not None:
                line += f", judge {summary.judge_score:.0%}"
            if summary.refusal_accuracy is not None:
                line += f", refused-unanswerable {summary.refusal_accuracy:.0%}"
            console.print(line + f"  -> {fname}")

    table = markdown_table(summaries, budget)
    console.print()
    console.print(Markdown(table))
    if answers:
        atable = answers_table(summaries)
        console.print(Markdown(atable))
    if model is not None and hasattr(model, "hits"):
        console.print(f"[dim]LLM cache: {model.hits} hits, {model.misses} calls[/]")
    if readme:
        ok = update_readme(readme, marker, table)
        if answers:
            ok = update_readme(readme, answers_marker, atable) and ok
        console.print(f"README {'updated' if ok else 'NOT fully updated (markers missing)'}: {readme}")


@eval_app.command("compare")
def eval_compare(
    a: Path = typer.Argument(..., help="Baseline results JSON"),
    b: Path = typer.Argument(..., help="Candidate results JSON"),
    metric: str = typer.Option("hit", help="hit | recall5 | mrr | judge"),
):
    """Paired, per-question comparison of two runs: is B really better than A, or is it noise?"""
    from .evaluation import load_results, paired_compare

    sa, ra = load_results(a)
    sb, rb = load_results(b)
    c = paired_compare(ra, rb, metric)
    console.print(f"[bold]{metric}[/] on {c['n']} paired questions")
    console.print(f"  A {sa['label']}: {c['mean_a']:.1%}")
    console.print(f"  B {sb['label']}: {c['mean_b']:.1%}")
    lo, hi = c["diff_ci"]
    console.print(f"  B - A = {c['mean_diff']:+.1%}  (95% CI {lo:+.1%} to {hi:+.1%})")
    console.print(f"  B better on {len(c['b_better'])}, worse on {len(c['b_worse'])}; sign test p = {c['p_sign_test']:.3f}")
    verdict = ("[green]likely a real improvement[/]" if c["mean_diff"] > 0 and c["p_sign_test"] < 0.05 else
               "[red]likely a real regression[/]" if c["mean_diff"] < 0 and c["p_sign_test"] < 0.05 else
               "[yellow]not distinguishable from noise at this sample size[/]")
    console.print(f"  -> {verdict}")
    if c["b_better"]:
        console.print(f"  better: {', '.join(c['b_better'])}")
    if c["b_worse"]:
        console.print(f"  worse:  {', '.join(c['b_worse'])}")


@eval_app.command("report")
def eval_report(results_file: Path):
    """Per-category breakdown and the list of retrieval failures (with each retriever's rank)."""
    from .evaluation import category_report, failure_report, load_results

    summary, results = load_results(results_file)
    console.print(f"[bold]{summary['label']}[/]")
    console.print(Markdown(category_report(results)))
    fails = failure_report(results)
    console.print(f"\n[bold]{len(fails)} question(s) whose evidence missed the context[/]")
    t = Table("Id", "Category", "Final rank", "Per-retriever rank", "Top results")
    for f in fails:
        t.add_row(f["id"], f["category"] or "", str(f["rank"]),
                  ", ".join(f"{k}:{v}" for k, v in f["retriever_ranks"].items()), "\n".join(f["top"]))
    console.print(t)


@eval_app.command("generate")
def eval_generate(
    index: str = typer.Option(..., "--index", "-i", help="Index to sample chunks from"),
    n: int = typer.Option(20, help="Questions to generate"),
    llm: str = typer.Option(None),
    out: Path = typer.Option(Path("evals/generated.json")),
    seed: int = typer.Option(7),
):
    """Draft questions (with reference answers) from random chunks. Review them by hand before trusting them."""
    model = _llm_or_exit(llm)
    store = open_store(index)
    pool = [c for c in store.chunks() if c.kind not in ("file",) and len(c.text) > 300]
    random.Random(seed).shuffle(pool)
    system = (
        "You write evaluation questions for a code search assistant. Given one code chunk, write ONE question a "
        "developer might ask that this chunk answers, without naming the exact method in the question when "
        "possible. Return JSON: {\"question\": str, \"contains\": str, \"answer_keywords\": [str], \"reference\": str} "
        "where \"contains\" is an exact substring copied from ONE line of the chunk that is the key evidence, "
        "answer_keywords are 1-3 short exact terms a correct answer must mention, and reference is a 1-2 sentence "
        "correct answer."
    )
    questions = []
    for c in pool[: n * 2]:
        if len(questions) >= n:
            break
        try:
            d = json.loads(model.complete(system, f"File: {c.path}\n\n{c.text}", json_mode=True))
        except (json.JSONDecodeError, RuntimeError) as e:
            console.print(f"[yellow]skip: {e}[/]")
            continue
        if d.get("contains") and d["contains"] in c.text:
            questions.append({
                "id": f"g{len(questions) + 1:02d}", "question": d["question"], "category": "generated",
                "expected": [{"path": c.path, "contains": d["contains"]}],
                "answer_keywords": d.get("answer_keywords", []), "reference": d.get("reference", ""),
            })
    meta = store.meta
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"repo": meta["source"], "ref": meta.get("commit"), "include": meta.get("include"),
                               "exclude": meta.get("exclude"), "questions": questions}, indent=2), encoding="utf-8")
    console.print(f"Wrote {len(questions)} draft questions to {out}. Review them, and add a few unanswerable "
                  "ones (\"answerable\": false), before use.")


if __name__ == "__main__":
    app()
