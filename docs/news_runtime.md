# News Runtime

News Runtime 通过环境变量选择 policy 和 retrieval provider：

```text
BANSO_NEWS_POLICY=rule_based|llm
BANSO_NEWS_RETRIEVAL_PROVIDER=tavily|local
```

默认使用 `rule_based` policy 和 Tavily。Local 模式只检索本地语料，并可通过
`BANSO_CORPUS_SEARCH_MODE=bm25|vector|hybrid` 选择索引模式，默认 `hybrid`。

Local 模式要求 corpus registry、SQLite 数据库和 LanceDB 索引已经存在；同步、
重建索引和路径配置见 [corpus.md](corpus.md)。BM25 不需要 embedding 配置。

运行 smoke check：

```bash
uv run python scripts/real_news_runtime_check.py
```
