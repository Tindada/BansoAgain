# AI professional news evaluation

`ai_professional_news.jsonl` is the initial live evaluation set for the real
news runtime. It covers model and product releases, research, benchmarks,
policy, safety, and industry events.

The cases intentionally do not contain fixed reference answers because the
queries are time-relative. The first evaluation stage measures objective
pipeline health only:

- how completely search result sources are classified;
- which source types and unknown domains appear;
- which route-specific research actions were executed;
- whether documents can be fetched and parsed;
- whether evidence and citations are produced;
- action latency and per-case failures.

`passed_minimums` does not mean that an answer is factually correct. It means
that the case met its minimum document, evidence, citation, and final-answer
requirements. Answer and citation quality require a later human or judge
evaluation stage.

The runtime always uses the LLM policy. Select `web`, `local`, or both retrieval
routes with `BANSO_NEWS_RETRIEVAL_ROUTES`. When both are enabled, the policy
selects the route for each atomic research action; there is no automatic route
fallback.

Run the full evaluation, for example with both routes enabled:

```bash
BANSO_NEWS_RETRIEVAL_ROUTES=local,web \
uv run python scripts/evaluate_news_runtime.py \
  --max-document-fetches 10 \
  --max-active-documents 6 \
  --output runs/eval_local_web.jsonl
```

Use `web` or `local` instead to isolate one route. Local evaluation requires a
prepared corpus database and index; `BANSO_CORPUS_SEARCH_MODE` optionally
selects `bm25`, `vector` (the default), or `hybrid`.

Use the same document budgets, models, extraction concurrency, and corpus index
when comparing runs. Run evaluations sequentially when they share an LLM
service so that latency measurements remain comparable.

Results are written incrementally to `runs/news_evaluation_<timestamp>.jsonl`,
so completed cases remain available if a later case fails or the process is
interrupted. A sibling `.traces.jsonl` stores the completed `SpanRecord` list for
each case that captured spans. `trace_id` links an evaluation result to that
list, and the enclosing JSON object contains the evaluation case ID. The
`.summary.json` records aggregate metrics, timestamps, output paths, the cumulative
and active document budgets, enabled retrieval routes, local search mode, and
configured model names. Aggregate research counts, active/shelved/unusable
document counts, curation actions, and action durations include every repeated
action. API keys are never written to these outputs.
