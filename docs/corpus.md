# 本地语料库运维

后台同步独立于 News Agent 运行。SQLite 保存最新文档正文，LanceDB 索引可随时
从 SQLite 重建。

## 配置

来源维护在 `config/trusted_sources.json`。所有记录都参与来源分类，只有
`enabled: true` 的来源参与后台摄取。可摄取来源必须明确列出允许的域名、正文路径
以及 RSS/Atom 或 Sitemap endpoint；新增来源前需要人工核实其归属和抓取范围。

默认数据路径为：

```text
data/corpus.sqlite3
data/corpus.lance
```

可以通过以下环境变量覆盖：

```text
BANSO_CORPUS_REGISTRY_PATH
BANSO_CORPUS_DATABASE_PATH
BANSO_CORPUS_INDEX_PATH
```

重建向量索引以及 vector/hybrid 检索还需要 OpenAI-compatible embedding 配置：

```text
BANSO_EMBEDDING_PROVIDER       # openai（默认）或 jina
BANSO_EMBEDDING_MODEL
BANSO_EMBEDDING_DIMENSIONS
BANSO_EMBEDDING_BASE_URL       # 可选
BANSO_EMBEDDING_API_KEY        # 取决于 endpoint
```

使用 Jina 托管的 `jina-embeddings-v5-text-small` 时可配置为：

```text
BANSO_EMBEDDING_PROVIDER=jina
BANSO_EMBEDDING_MODEL=jina-embeddings-v5-text-small
BANSO_EMBEDDING_DIMENSIONS=1024
BANSO_EMBEDDING_BASE_URL=https://api.jina.ai/v1
BANSO_EMBEDDING_API_KEY=<Jina API key>
```

Jina provider 为索引文档发送 `retrieval.passage` task，为检索查询发送
`retrieval.query` task；返回向量仍使用配置的维度进行校验。官方托管 API 需要
配置上述 base URL 和 API key。

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

## 待完成：白名单直达与 HTML 索引发现

部分官方博客没有可用的 RSS/Atom 或 Sitemap，但存在稳定的文章 URL 或 HTML 栏目页。
后续可在来源配置中增加两类受控发现入口：

- `document_urls`：将少量、明确列入白名单的 URL 本身作为文档摄取，不从其内容继续发现链接；
- `html_index_urls`：从栏目页提取一层文章链接，不递归跟随文章内链接。

HTML 索引发现需要保存 endpoint validator 和已发现 URL，并限制每轮解析页数、候选
链接数与新增文档数。日常同步默认只处理栏目首页的近期内容；历史回填应作为显式、
有上限的操作。所有 document、index、跳转后 URL 和发现 URL 仍需通过来源域名、路径与
robots 校验，不以伪装 User-Agent 等方式绕过站点限制。

Meta AI 是首个候选用例：`https://ai.meta.com/blog/` 与
`https://ai.meta.com/research/` 可作为未来的 HTML index；在该能力完成前，这两个
路径只参与来源分类和摄取范围声明，不启用自动链接发现。

xAI 主站的 `robots.txt` 虽声明 `https://x.ai/sitemap.xml`，但 sitemap、新闻栏目和
文章页面当前均对 corpus User-Agent 返回 HTTP 403；其 Content Signal 还声明
`ai-input=no`，因此保持禁用且不尝试绕过。`data.x.ai` 上的公开模型卡和安全框架 PDF
可以直接获取，已作为独立的 disabled 分类来源保留，待 `document_urls` 支持后再逐份审核。

PMLR 的 `https://proceedings.mlr.press/feed.xml` 只发现卷页面；直接摄取会把整卷作为一个
文档，导致检索结果缺少论文级标题、URL 与 provenance，因此保持禁用，待
`html_index_urls` 支持从卷页面发现单篇论文后再启用。

OpenReview 没有可用的 RSS 或 Sitemap，官方发现方式是 Notes API；当前 API、forum 页面
和 PDF 均要求 challenge verification，无法由 corpus runtime 稳定访问，因此保持禁用且
不尝试绕过。未来需在站点允许机器访问后增加专用 API discovery，不能由通用 HTML index
替代，因为提交、评审和回复由不同类型的 Note 表示。
