# Banso

Banso is an experimental news research agent. An LLM policy chooses atomic
research actions, each of which retrieves search results, fetches documents,
and extracts evidence. The agent can then curate the evidence-bearing documents
and synthesize an answer with citations.

## Setup

Banso requires Python 3.12 and uses [uv](https://docs.astral.sh/uv/) for project
and dependency management.

```bash
uv sync
cp .env.template .env
```

Configure the policy and extraction model under `VLLM_*`, and the answer
synthesis model under `EXTERNAL_LLM_*`. Both accept OpenAI-compatible endpoints
and may point to either local or hosted model services.

The default `web` route uses
[Tavily](https://docs.tavily.com/documentation/api-reference/introduction), a
hosted Web search API, to retrieve result URLs and snippets. It requires a
separate `BANSO_TAVILY_API_KEY`; Banso fetches the source documents itself.

## Run the news agent

The root `main.py` is the normal application entry point:

```bash
uv run python main.py "What important AI models were released recently?"
```

Optional query context can be supplied explicitly:

```bash
uv run python main.py \
  "What changed in AI regulation this month?" \
  --language en \
  --region "united states" \
  --time-range month
```

Set `BANSO_NEWS_RETRIEVAL_ROUTES` in `.env` to `web`, `local`, or `local,web`.
With both routes enabled, the policy chooses a route for each research action.
See [News Runtime](docs/news_runtime.md) for details.

## Local corpus

The local route searches a trusted-source corpus. Synchronize its sources and
rebuild the derived index before enabling it:

```bash
uv run banso-corpus sync
uv run banso-corpus reindex
```

Corpus sources, storage, indexing modes, and maintenance are documented in
[Local News Corpus](docs/corpus.md).

## Development

Run the test suite:

```bash
uv run pytest
```

The scripts under `scripts/` are development and evaluation tools rather than
the normal application entry point. See [News Evaluation](evaluations/README.md)
for the live evaluation workflow and [Project Plan](docs/plan.md) for the
current implementation status.

## Third-party content

Banso does not distribute documents fetched from third-party sources. Content
retrieved at runtime remains subject to the rights and terms of its original
publisher.

## License

Banso is licensed under the [MIT License](LICENSE).
