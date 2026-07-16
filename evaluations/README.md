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
- whether documents can be read;
- whether evidence and citations are produced;
- action latency and per-case failures.

`passed_minimums` does not mean that an answer is factually correct. It means
that the case met its minimum document, evidence, citation, and final-answer
requirements. Answer and citation quality require a later human or judge
evaluation stage.

Run a low-cost two-case check first:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_news_runtime.py --limit 2
```

Run the full set:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_news_runtime.py
```

Results are written incrementally to `runs/news_evaluation_<timestamp>.jsonl`,
so completed cases remain available if a later case fails or the process is
interrupted. A sibling `.traces.jsonl` stores the complete `AgentTrace` for each
successful case; `trace_id` links an evaluation result to its trace, and the
trace metadata contains the evaluation case ID. The `.summary.json` records
aggregate metrics, timestamps, output paths, the document budget, and configured
model names. Aggregate search counts and action durations include every repeated
action. API keys are never written to these outputs.
