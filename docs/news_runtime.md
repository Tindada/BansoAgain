# News Runtime

News Runtime 始终使用 LLM policy。通过环境变量选择 research action 的粒度：

```text
BANSO_NEWS_POLICY=atomic|search_read
```

默认值为 `search_read`，policy 分别执行 `SEARCH` 和 `READ`：可先进行多次搜索，
再从累积的候选结果中选择文档读取。`atomic` 是备选 policy，每次通过一个原子的
`RESEARCH` action 完成 retrieval、结果选择、document fetch 和 evidence extraction。

通过环境变量显式启用 routes：

```text
BANSO_NEWS_RETRIEVAL_ROUTES=web|local|local,web
```

默认值为 `web`。`web` 使用 Tavily retrieval 和 HTTP fetch；`local` 使用本地语料
retrieval 和 corpus-aware fetch。启用 `local,web` 时两路都会初始化，由 LLM 在每次
`RESEARCH` 或 `SEARCH` 中选择一路；runtime 不进行跨 route fallback。

远程 document fetcher 通过以下变量选择：

```text
BANSO_DOCUMENT_FETCHER=http|jina
BANSO_JINA_API_KEY=                       # 可选
```

默认仍为 `http`。选择 `jina` 后，Web route 和 local corpus 未命中时的 fallback
都会使用同一个 Jina Reader fetcher；它直接采用 Reader 返回的 Markdown，不再调用
`DocumentParser`。API key 未配置时不发送 `Authorization`，可使用 Jina 的匿名入口，
其[官方 Reader 文档](https://jina.ai/reader/)当前列出的限制为 20 RPM。

调用遵循官方的 URL 前缀形式，并显式请求 JSON 和绕过缓存：

```http
GET https://r.jina.ai/https://example.com/article
Accept: application/json
X-No-Cache: true
Authorization: Bearer jina_...  # 仅配置 key 时发送
```

请求头与 GET/POST 能力也可在 Jina Reader 的
[官方参数定义](https://github.com/jina-ai/reader/blob/main/src/dto/crawler-options.ts)
和[官方服务实现](https://github.com/jina-ai/reader/blob/main/src/api/crawler.ts)
中核对。

Local route 可通过 `BANSO_CORPUS_SEARCH_MODE=bm25|vector|hybrid` 选择索引模式，
默认 `vector`。Local route 要求 corpus registry、SQLite 数据库和 LanceDB 索引已经
存在；同步、重建索引和路径配置见 [corpus.md](corpus.md)。BM25 不需要 embedding
配置。

运行 smoke check：

```bash
uv run python scripts/real_news_runtime_check.py
```
