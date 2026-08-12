# News Runtime

News Runtime 始终使用 LLM policy。Policy 每次通过一个原子的 `RESEARCH` action
同时选择 query 和 retrieval route；runtime 在 action 内完成 retrieval、结果选择、
document fetch 和 evidence extraction。

通过环境变量显式启用 routes：

```text
BANSO_NEWS_RETRIEVAL_ROUTES=web|local|local,web
```

默认值为 `web`。`web` 使用 Tavily retrieval 和 HTTP fetch；`local` 使用本地语料
retrieval 和 corpus-aware fetch。启用 `local,web` 时两路都会初始化，由 LLM 在每次
`RESEARCH` 中选择一路；runtime 不进行跨 route fallback。

Local route 可通过 `BANSO_CORPUS_SEARCH_MODE=bm25|vector|hybrid` 选择索引模式，
默认 `vector`。Local route 要求 corpus registry、SQLite 数据库和 LanceDB 索引已经
存在；同步、重建索引和路径配置见 [corpus.md](corpus.md)。BM25 不需要 embedding
配置。

运行 smoke check：

```bash
uv run python scripts/real_news_runtime_check.py
```
