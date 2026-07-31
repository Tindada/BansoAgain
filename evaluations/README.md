# AI professional news evaluation

`ai_professional_news.jsonl` is the initial live evaluation set for the real
news runtime. It covers model and product releases, research, benchmarks,
policy, safety, and industry events.

The cases intentionally do not contain fixed reference answers because the
queries are time-relative. The first evaluation stage measures objective
pipeline health only:

- how completely search result sources are classified;
- which source types and unknown domains appear;
- which searches were planned and executed;
- whether documents can be fetched and parsed;
- whether evidence and citations are produced;
- action latency and per-case failures.

`passed_minimums` does not mean that an answer is factually correct. It means
that the case met its minimum document, evidence, citation, and final-answer
requirements. Answer and citation quality require a later human or judge
evaluation stage.

Run a low-cost two-case check first:

```bash
BANSO_NEWS_POLICY=llm uv run python scripts/evaluate_news_runtime.py --limit 2
```

Run both policies with their respective document budgets. The rule-based policy
uses matching cumulative and active limits because it does not perform evidence
curation; the LLM policy uses a smaller active working set than its cumulative
fetch budget to exercise curation:

```bash
BANSO_NEWS_POLICY=rule_based uv run python scripts/evaluate_news_runtime.py --max-document-fetches 6 --max-active-documents 6
BANSO_NEWS_POLICY=llm uv run python scripts/evaluate_news_runtime.py --max-document-fetches 10 --max-active-documents 6
```

Run the full set:

```bash
BANSO_NEWS_POLICY=llm uv run python scripts/evaluate_news_runtime.py
```

Set `BANSO_NEWS_POLICY` to `llm` for the LLM policy or `rule_based` for the
fixed-flow baseline.

`BANSO_NEWS_RETRIEVAL_PROVIDER` selects an isolated retrieval path: `tavily`
keeps the v4 baseline behavior and is the default, while `local` only searches
the trusted corpus. Local runs can select `bm25`, `vector`, or `hybrid` through
`BANSO_CORPUS_SEARCH_MODE`; the default is `hybrid`. Run Tavily-only and
local-only evaluations separately. Cross-provider fallback and ranking are not
implemented yet.

Results are written incrementally to `runs/news_evaluation_<timestamp>.jsonl`,
so completed cases remain available if a later case fails or the process is
interrupted. A sibling `.traces.jsonl` stores the completed `SpanRecord` list for
each case that captured spans. `trace_id` links an evaluation result to that
list, and the enclosing JSON object contains the evaluation case ID. The
`.summary.json` records aggregate metrics, timestamps, output paths, the cumulative
and active document budgets, and configured model names. Aggregate search counts,
active/shelved/unusable document counts, curation actions, and action durations
include every repeated action. API keys are never written to these outputs.
