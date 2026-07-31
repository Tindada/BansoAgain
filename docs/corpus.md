# 本地语料库运维

本地语料库独立于 News Agent 运行。SQLite 保存最新文档正文，LanceDB 索引可随时
从 SQLite 重建。

## 配置

官方来源维护在 `config/trusted_sources.json`。来源必须明确列出允许的域名、正文
路径以及 RSS/Atom 或 Sitemap endpoint；新增来源前需要人工核实其归属和抓取范围。

默认数据路径为：

```text
data/corpus.sqlite3
data/corpus.lance
```

可以通过以下环境变量覆盖：

```text
BANSO_CORPUS_REGISTRY
BANSO_CORPUS_DATABASE
BANSO_CORPUS_INDEX
```

重建向量索引以及 vector/hybrid 检索还需要 OpenAI-compatible embedding 配置：

```text
BANSO_EMBEDDING_MODEL
BANSO_EMBEDDING_DIMENSIONS
BANSO_EMBEDDING_BASE_URL       # 可选
BANSO_EMBEDDING_API_KEY        # 可选
```

## 运行

同步所有启用的来源：

```bash
uv run banso-corpus sync
```

从 SQLite 重建 BM25、向量和混合检索共用的索引：

```bash
uv run banso-corpus reindex
```

检索时可以逐次选择模式：

```bash
uv run banso-corpus search "agentic AI" --mode hybrid --limit 5
uv run banso-corpus search "agentic AI" --mode vector --limit 5
uv run banso-corpus search "agentic AI" --mode bm25 --limit 5
```

BM25 检索不调用 embedding provider，因此只使用已有索引时无需 embedding 环境变量。
命令输出为 JSON；`sync` 遇到任一可恢复抓取失败时仍会完成其他来源，但退出码为 1，
并在输出中列出失败 URL 和原因。
