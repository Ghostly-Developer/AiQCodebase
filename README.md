# Ask My Codebase

A retrieval-augmented (RAG) assistant that indexes a Git repository and answers questions about it,
citing the **file and line range** behind every claim, with clickable links to the exact lines on GitHub.

Output format (illustrative; the cited line numbers are real for the pinned Dapper commit):

```
$ askcode ask "Why are SingleResult and SingleRow disabled by default?" -i dapper
They are excluded from the allowed command behaviours by default so that errors raised by the
database *after* the SELECT are still detected properly [1]. If a provider throws because it
rejects them, Dapper turns both off and retries without them [2].

Sources
  [1] Dapper/SqlMapper.Settings.cs:L14-L15   https://github.com/DapperLib/Dapper/blob/1880947…/Dapper/SqlMapper.Settings.cs#L14-L15
  [2] Dapper/SqlMapper.Settings.cs:L42-L55   https://github.com/DapperLib/Dapper/blob/1880947…/Dapper/SqlMapper.Settings.cs#L42-L55
```

It is built to be *measured*: a 62-question eval set (50 with known answers, 12 deliberately
unanswerable) scores every chunking strategy and retrieval mode automatically, with confidence intervals.

## Results

Indexed: [DapperLib/Dapper](https://github.com/DapperLib/Dapper) at commit `1880947` (86 files, about 12k lines of
C# plus docs; tests excluded). Embeddings: local `BAAI/bge-small-en-v1.5` unless noted. Questions:
[`evals/dapper.json`](evals/dapper.json). Ranges in brackets are 95% bootstrap confidence intervals over the 50 answerable questions.

### Chunking strategies × retrieval modes

<!-- EVAL:START -->
_Run `askcode eval run evals/dapper.json --readme README.md` to fill this table._
<!-- EVAL:END -->

ANALYSIS_PLACEHOLDER

### Answer quality (LLM judge)

<!-- ANSWERS:START -->
_Needs an API key. Run:_
`askcode eval run evals/dapper.json --answers --llm groq --chunkers ast --modes hybrid,hybrid+expand --readme README.md`
<!-- ANSWERS:END -->

This table reports the judge score against expert reference answers, keyword accuracy, the share of answers that are
correct *and* cite the evidence lines, how often the 12 unanswerable questions were correctly declined, false refusals,
and the rate of invalid citations and invented identifiers per answer.

**Metric definitions**

| Metric | Meaning |
|---|---|
| Recall@k | The chunk containing the evidence line is in the top *k* results |
| File@5 | The right *file* is in the top 5. Whole-file chunking always looks good here, which is why line-level recall matters |
| MRR@10 | Mean of 1/rank of the first correct chunk |
| Hit@2000tok | After packing the ranked results into a 2,000-token prompt, the evidence line is still in it. This is closest to "could the LLM answer?" |
| Judge score | An LLM grades each answer against the reference answer: correct = 1, partial = 0.5 |
| Refused unanswerable | Share of the 12 questions about features Dapper doesn't have that were declined instead of answered |
| Correct + cited | Keyword-correct **and** a verified citation within 5 lines of the evidence |

## How it works

Component diagram, flow and sequence diagrams, storage schema and pseudo code for each algorithm:
**[ARCHITECTURE.md](ARCHITECTURE.md)**.

```
repo ──► loader (secret filter + redaction) ──► chunker ──► [LLM chunk summaries] ──► embedder ──► store
                                                file│fixed│function│ast                            local numpy | Postgres+pgvector
question ─► [condense follow-up] ─► [rewrite: keywords + hypothetical code]
         ─► vector search ─┐
         ─► HyDE search ───┼─► RRF fusion ─► [cross-encoder rerank] ─► pack into token budget [small-to-big]
         ─► BM25 search ───┘                                                     │
                       Gemini / Groq ◄── numbered, line-numbered sources ◄───────┘
                            │ (streamed)
                            └─► answer with [S2:L120-L131] ─► citation check ─► answer + GitHub links + trace log
```
Steps in [brackets] are optional.

| Piece | File | Notes |
|---|---|---|
| Loading | [loader.py](src/askcode/loader.py), [safety.py](src/askcode/safety.py) | Local path or git URL; `--ref` pins a commit; `.env`/keys skipped, inline secrets redacted |
| Chunking | [chunking.py](src/askcode/chunking.py) | `file`, `fixed` (≈400-token windows), `function` (brace scanner / Python `ast`), `ast` (tree-sitter, 160+ languages) |
| Contextual retrieval | [contextualize.py](src/askcode/contextualize.py) | `--contextualize`: an LLM writes a one-line summary per chunk, embedded with it |
| Embeddings | [embeddings.py](src/askcode/embeddings.py) | `fastembed` (local), `fastembed:jinaai/jina-embeddings-v2-base-code`, `google`; disk-cached |
| Vector store | [store.py](src/askcode/store.py) | `local` (numpy + JSONL) or `pgvector` (HNSW cosine + Postgres full-text) |
| Keyword search | [bm25.py](src/askcode/bm25.py) | BM25 with a code tokenizer (`QueryFirstOrDefault` → query/first/default) |
| Retrieval | [retrieval.py](src/askcode/retrieval.py), [rewrite.py](src/askcode/rewrite.py) | RRF hybrid, HyDE + keyword query rewriting, configurable cross-encoder, small-to-big packing |
| Answers | [answer.py](src/askcode/answer.py), [pipeline.py](src/askcode/pipeline.py), [llm.py](src/askcode/llm.py) | Gemini or Groq, streamed, disk-cached; citations verified; follow-ups condensed |
| Web UI | [server.py](src/askcode/server.py), [static/index.html](src/askcode/static/index.html) | FastAPI + SSE streaming, chat history, expandable sources, GitHub links |
| Eval | [evaluation.py](src/askcode/evaluation.py), [judge.py](src/askcode/judge.py) | Recall/MRR/Hit@budget, LLM judge, refusal rate, bootstrap CIs, paired sign test, failure report |

### Reducing hallucinations

1. **Grounded prompt.** The model sees only the retrieved sources, each with line numbers, and must cite
   `[S<n>:L<a>-L<b>]` after every factual sentence, or reply *"I don't know based on the indexed code."*
2. **Citation verification.** Each citation is checked: does `S<n>` exist, and are the lines inside it? Invalid ones are shown as *unverified*.
3. **Unsupported identifier check.** Any `identifier` in backticks that appears in no source is flagged.
4. **Uncited answers are flagged** as unverified.
5. **Measured, not assumed.** The 12 unanswerable questions measure how often the assistant declines instead of inventing an API.
6. **Prompt-injection guard.** Repository text is untrusted. Sources are wrapped in tags, and the model is told never to follow instructions inside them.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -e ".[dev]"         # add ,pgvector for the Postgres backend
copy .env.example .env          # then add GOOGLE_API_KEY and/or GROQ_API_KEY
```

Indexing, search and the retrieval eval need **no API key**. Embeddings run locally; models download to
`.askcode/models` on first use. An API key is needed for answers, `chat`, `serve`, `--rewrite`, `--contextualize`
and `eval run --answers`.

## Usage

```bash
# Index a repo (local path or git URL). Default chunker: ast (tree-sitter)
askcode index . --name mine
askcode index https://github.com/DapperLib/Dapper.git --name dapper \
    --ref 188094784256da61d7a37c43ea54c64213c2ce7c --exclude "tests/*" --exclude "benchmarks/*"
askcode index . --name mine-ctx --contextualize --llm groq          # contextual retrieval
askcode update mine                                                 # re-embed only changed files

# Retrieval only (no LLM)
askcode search "how is the command timeout chosen" -i dapper --trace

# Answers
askcode ask "How do I map snake_case columns to properties?" -i dapper
askcode ask "..." -i dapper --llm groq --rewrite --rerank --show-sources --trace traces.jsonl
askcode chat -i dapper                                              # follow-up questions in the terminal
askcode serve                                                       # web UI at http://127.0.0.1:8000

# Evaluation
askcode eval check evals/dapper.json
askcode eval run evals/dapper.json --chunkers file,fixed,function,ast --modes vector,keyword,hybrid,hybrid+expand
askcode eval run evals/dapper.json --answers --llm groq --modes hybrid,hybrid+expand,hybrid+rewrite+expand
askcode eval compare evals/results/A.json evals/results/B.json --metric hit    # is B really better?
askcode eval report evals/results/A.json                                       # per-category + failures
askcode eval generate -i mine --n 30 --out evals/mine.json                    # draft questions for your repo
```

Mode strings for `eval run --modes` combine a base (`vector`, `keyword`, `hybrid`) with flags:
`+expand` (small-to-big), `+rewrite` (LLM query rewriting), `+rerank` or `+rerank:<model>`.

| Option | Values |
|---|---|
| `--chunker` | `ast` (default), `function`, `fixed`, `file` |
| `--embedder` | `fastembed` (bge-small), `fastembed:jinaai/jina-embeddings-v2-base-code`, `google`, `google:<model>` |
| `--store` | `local` (default) or `pgvector` |
| `--llm` | `google`, `groq`, `google:<model>`, `groq:<model>`. Default: whichever key is set. Models: `ASKCODE_GOOGLE_MODEL`, `ASKCODE_GROQ_MODEL` |
| `--reranker-model` | any fastembed cross-encoder, e.g. `jinaai/jina-reranker-v1-turbo-en` |

### pgvector

```bash
docker compose up -d                  # Postgres 17 + pgvector on :5432
pip install -e ".[pgvector]"
askcode index <repo> --name dapper-pg --store pgvector
askcode ask "..." -i dapper-pg
```

Chunks get an HNSW index (`vector_cosine_ops`) and a generated `tsvector` column with a GIN index for the keyword side;
full file text is stored for small-to-big expansion. CI runs the pgvector integration test against a
`pgvector/pgvector:pg17` service container ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

### Caching, traces and cost

- LLM responses and embeddings are cached in `.askcode/cache/*.sqlite`. Re-running an eval costs nothing and returns the same answers.
- `--trace file.jsonl` (on `ask`, `chat`, `serve`) logs each query: the rewritten query, every retriever's top-10, final ranks, sources, answer, citations and timings.
- `eval report` lists every question whose evidence missed the context, with the rank each retriever gave it. This shows whether the miss came from vector search, keyword search or fusion.

## Eval set format

[`evals/dapper.json`](evals/dapper.json) pins the repo and commit. Each question names its evidence by
**file plus a literal snippet** (resolved to a line at eval time), and carries a reference answer for the judge:

```json
{"id": "q19", "category": "config",
 "question": "How do I make Dapper map snake_case columns like first_name to a FirstName property?",
 "expected": [{"path": "Dapper/DefaultTypeMap.cs", "contains": "public static bool MatchNamesWithUnderscores { get; set; }"}],
 "answer_keywords": ["MatchNamesWithUnderscores"],
 "reference": "Set DefaultTypeMap.MatchNamesWithUnderscores = true so underscores are ignored when matching column names to members."}
{"id": "u01", "category": "unanswerable", "answerable": false,
 "question": "How do I configure Dapper's Redis-backed second-level cache for query results?"}
```

`askcode eval check` fails if a snippet no longer matches. Each unanswerable question was checked by grepping the
pinned source to confirm the feature doesn't exist.

## Project layout

```
src/askcode/        library, CLI (askcode = askcode.cli:app) and web UI
evals/              eval sets; results/ holds per-question JSON for every run
tests/              unit + end-to-end tests (offline: hash embedder, fake LLM); pgvector test runs in CI
.github/workflows/  CI: tests on 3.11/3.13 with a pgvector service, eval-set check
```
