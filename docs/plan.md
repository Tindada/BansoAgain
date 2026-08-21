# News Agent 总体计划

## 背景

目标是设计并逐步实现一个“新闻搜索 + 信息筛选 + 总结”的新闻 Agent。当前已完成
核心 Runtime、LLM Policy 所需的 State、Artifact 与 Policy Context、原子化
`RESEARCH(query, route)`、跨步骤 URL 去重和独立 LLM tracing。真实运行
入口始终使用 `LLMNewsPolicy`，并可同时启用 Web 与 Local retrieval routes。

系统需要长期支持：

- 可替换 LLM
- 可替换搜索与 retrieval
- 可基于 agent rollout 和 reward 对 LLM policy 进行 RL 后训练
- 完整执行轨迹记录
- 可评估、可回放、可迭代优化

核心原则：

> 核心 runtime 自研，外部能力插件化接入。

## 当前状态

已完成：

- 已初始化 Python 项目结构，并使用 `uv` 管理依赖。
- 已定义核心 runtime、state、action、policy、executor、reducer 等基础模块；
  `AgentResult` 和 `RuntimeRunResult` 已集中到 runtime 模块。
- 已实现最小 `AgentRuntime` 主循环，支持 policy 决策、executor 执行、reducer 更新状态和 trace 收集。
- 已将新闻研究流程合并为原子的 `RESEARCH(query, route)` action，在 action 内完成
  retrieval、结果选择、document fetch 和 evidence extraction。
- 已实现最小 `LLMNewsPolicy`，由 LLM 根据有界 Policy Context、可用 action 和剩余
  预算选择结构化 `AgentAction`，并校验动作参数与 research 预算；Action
  availability 由资源生命周期和剩余额度决定。
- `LLMNewsPolicy` 的非法输出和已知 LLM 调用失败会作为带 reason 的 policy error
  向上抛出，由 Runtime Span 保存失败信息；Policy 选择失败不在 Policy
  内重试或自动回退。
- 真实新闻运行入口始终使用 LLM Policy；通过 `BANSO_NEWS_RETRIEVAL_ROUTES`
  显式启用 `web`、`local` 或两路，由 LLM 为每次 research 选择 route。
- 已通过 provider-independent 的 `TracingLLMClient` 统一记录 LLM 实际输入、原始
  provider 响应、completion 和 token usage；业务解析结果继续保存在对应
  Observation、Artifact 或外层 Span 中。
- 已实现可替换的 retrieval、document fetcher、evidence extractor、synthesizer 和 LLM client 接口及部分实现。
- 已接入 Tavily retrieval、HTTP document fetcher、OpenAI SDK LLM client、LLM evidence extractor 和 LLM synthesizer。
- HTTP document fetcher 会在解析正文前校验响应 Content-Type，支持 HTML/XHTML
  与带文本层的 PDF，其他类型作为明确的文档获取失败记录；HTML/PDF 解析已提取为
  可复用的 `DocumentParser`，供后续后台摄取链路使用。
- 已完成官方来源注册表、RSS/Atom 与 Sitemap 发现、robots 校验、条件请求、
  HTML/PDF 解析和 `SQLiteCorpusStore` 写入编排；该链路仍独立于 Agent。
- 已新增段落感知分块和可从 SQLite 重建的 LanceDB 本地索引，支持按次选择
  BM25、向量或混合检索；真实 News Runtime 可同时启用本地语料和 Web route。
- 已实现超长文档的分块 evidence extraction，并隔离单篇文档的 LLM 提取失败。
- LLM evidence extraction 为单篇文档设置最大 chunk 数，超过上限时在调用 LLM 前
  将该文档记录为 `document_too_large`，避免异常文档无上限占用执行时间。
- 已实现 retrieval filter、仅补充元数据而不做准入的 search result source
  classifier，以及基础 artifact store。
- Search 结果和文档已通过 State 中的标准化 URL 索引实现去重，同一 URL 复用首次
  保存的 artifact；单次 Search 内部的重复结果继续由 retrieval filter 过滤。
- Retrieval filter 仅允许可由当前文档获取链路直接消费的绝对 HTTP(S) URL，非法
  URL 会在保存 artifact 前被丢弃并记录分类计数。
- 已实现与业务模型解耦的 `SpanRecord`、`Tracer` 和 `InMemoryTraceSink`，通过
  `ContextVar` 传播当前 Span，并以 `trace_id` 关联运行结果和执行轨迹。
- `AgentState` 是运行进度的权威记录，保存 Action/Observation 历史、artifact ID、
  URL 索引、最终答案和 citations，并在初始化时固定本次运行使用的 UTC
  `reference_time`；完整 artifact 继续由 `ArtifactStore` 保存。
- State 分别记录每个 Search Result 的文档获取进度，以及每个 Document 的证据提取进度、
  Evidence ID 和可选的 active/shelved/unusable 生命周期；未完成提取时生命周期为
  `None`，成功形成 Evidence 后成为 active，空证据或终态失败成为 unusable，
  shelved 只来自 Agent 精筛。
- 内存 ArtifactStore 已保证同 ID 不可覆盖，并在写入、读取和列举时提供隔离快照。
- 已实现 `ResearchContextBuilder`，从 State 和 ArtifactStore
  确定性构造用户查询、参考时间、剩余预算、搜索历史、资源生命周期摘要以及有界的
  Search Result、Document 和分组 Evidence 预览，并显式标记省略数量。
- `reference_time` 在运行初始化时固定，并通过 Policy Context 为“最近”“本周”
  等相对时间提供统一基准。
- Runtime 已分别使用 Policy、Executor 和 Reducer 子 Span 记录耗时；失败 Span
  记录异常类型和信息，Trace 自身失败不会改变业务执行结果。
- 已补充覆盖核心 runtime、新闻执行器、retrieval、document fetcher、LLM 配置和 LLM 组件的测试。
- 早期固定流程与 v4 LLM evaluation 已作为历史 baseline 归档；当前评估以
  原子 Research 结构下的 Web 和 Local/Web runs 为准。

## 阶段 1：系统架构设计

目标：明确整个 agent 系统的边界、模块和数据流。

产出：

- 系统整体架构图
- 核心模块划分
- Agent 执行流程
- 未来基于 rollout 的 LLM RL 后训练扩展点说明

重点模块：

```text
AgentRuntime
Policy
AgentState
AgentAction
RetrievalProvider
DocumentFetcher
DocumentRanker
EvidenceExtractor
Synthesizer
Tracer
TraceSink
RolloutStore
RewardModel
LLMPolicyTrainer
```

## 阶段 2：核心接口设计

目标：先定义稳定接口，不绑定具体实现。

产出：

- `AgentState` 数据结构
- `AgentAction` 动作空间
- `Policy` 接口
- `LLMProvider` 接口
- `RetrievalProvider` 接口
- `Tracer` 和 `TraceSink` 接口
- `RolloutRecord`、`RewardModel` 和 `LLMPolicyTrainer` 预留接口

设计原则：

```text
可替换 LLM
可替换搜索/检索
可替换 policy
可记录完整 trace
可支持未来对 LLM policy 进行 RL 后训练
```

## 阶段 3：最小 Agent Runtime

目标：实现一个最小可运行状态机。

核心流程：

```text
接收 query
初始化 state
policy 选择 action
executor 执行动作
reducer 更新 state
tracer 记录运行边界 Span
循环直到 FINISH 或 STOP
```

第一版不追求复杂智能，先保证主循环清晰、可测试、可扩展。

## 阶段 4：新闻搜索链路 MVP

目标：实现最小新闻搜索与总结流程。

包含：

- Query 分析
- 生成搜索 query
- 调用搜索 provider
- 获取文档
- 筛选相关文档
- 抽取 evidence
- 多源总结
- 输出最终新闻摘要

## 阶段 5：LLM Agent Policy

目标：在保持 runtime 和 executor 边界不变的前提下，让 LLM 根据当前 state、
已有 observation 和剩余预算，从受约束的动作空间中动态选择下一步 action。

LLM policy 使用受约束的动作空间：

```text
RESEARCH
CURATE_EVIDENCE
FINISH
STOP
```

核心要求：

- `RESEARCH(query, route)` 在内部完成 retrieval、单次 search-result selection、
  fetch 和 extraction；LLM 不在这些内部阶段之间做选择。
- 允许 Policy 重复或改写 research query；SearchResult 和 Document 仍通过运行内稳定的
  URL 索引去重。fetch/extraction 的可恢复错误由 executor 在当前 action 内有界
  重试，不进入 LLM policy 或跨 action lifecycle。
- `FINISH` 生成最终答案并终止运行；`STOP` 不生成新答案，直接终止运行。
- `CURATE_EVIDENCE` 允许 LLM 根据相关性、信息增量、重复程度、覆盖缺口和来源质量，
  在已有 Evidence 的文档组之间进行 active/shelved 精筛；搁置不会删除产物或返还
  累计获取预算。空证据及终态提取失败文档自动成为不可恢复的 unusable，不计为
  Agent 精筛行为。
- Policy Context 使用 rollout 内稳定的 `D1`、`D2` 短引用供 LLM 选择文档，Policy
  校验后将引用转换为内部 UUID，LLM 不直接接触 Artifact ID。LLM 通过
  `active_document_refs` 声明精筛后的完整 active 集合，Policy 根据当前生命周期
  推导内部的 shelve/reactivate 差异，避免让模型负责状态转换方向。
- Policy Context 显示 active、shelved、unusable 状态及 active 超限数量；超限时
  `FINISH` 暂不可用，但 `STOP` 仍作为模型可自主选择的放弃作答动作保留。
- 输出可校验的结构化 `AgentAction`，不允许生成任意工具调用。
- `LLMNewsPolicy` 内部使用 `ResearchContextBuilder`，根据 State 和 ArtifactStore
  构造模型可见的决策事实；通用 `Policy` 接口继续只返回 `AgentAction`。
- 向模型提供 query、预算、语义化执行历史、有界 artifact context 和可用 action，
  不直接暴露完整 State，也不在 State 中重复保存 artifact summary。
- 对非法 action、无效参数和 LLM 调用失败提供确定性校验和明确错误；
  是否增加纠正重试由 evaluation 结果决定。
- 记录 action 选择所需的简短 decision metadata，保证行为可审计。
- 早期固定流程 policy 仅作为历史实现保留，不接入真实运行入口。
- 累计文档获取预算只约束搜索可用性和获取执行边界；active 文档上限只约束精筛后的
  Evidence 工作集及 `FINISH`。最大步骤数、搜索数和 token/cost 预算继续独立生效。

当前已完成最小 Policy、资源生命周期约束、精简的显式 Policy Context、确定性输出
校验、独立 LLM tracing、真实运行入口和 action-local fetch/extraction 重试；
token/cost 预算、retrieval 重试和是否为 Policy 增加纠正重试仍属于后续工作。

评估重点：

```text
任务完成率
无效动作率
搜索、文档、evidence 和 citation 数量
答案质量
执行延迟
token 与调用成本
相对固定流程 policy 的收益
```

实施顺序：先解决会导致整次运行失败的硬阻塞，并具备保存异常和 partial trace
的最低可审计能力；随后立即实现 LLM Agent Policy，不要求先清空全部已知问题。

## 阶段 6：Trace 与评估系统

目标：为 LLM Agent Policy 的评估、迭代优化和未来 RL 后训练做准备。

需要记录：

```text
用户输入
每一步 state
每一步 action
policy 实际接收的 prompt/messages 和可用 action schema
LLM 原始输出、结构化 action 解析结果和校验/重试记录
工具调用输入
工具调用输出
中间判断
最终答案
episode 终止原因和 outcome metrics
模型、prompt、schema、配置与代码版本
用户反馈
自动评分
```

此前 Trace 由 Runtime 持有，并绑定 `AgentTrace`、`TraceStep` 等业务模型，因此只能
记录 Runtime 直接可见的 State、Action、Observation、耗时和失败信息；LLM、Retrieval
等深层组件产生的观测数据无法在不修改业务参数和返回值的情况下关联到同一次运行。

当前 Trace 已重构为通用 Span：业务数据继续通过显式参数和返回值传递，`ContextVar`
只传播当前 Span，组件通过 `Tracer` 记录观测数据并由 `TraceSink` 独立收集。Runtime
只声明 `agent.run`、`agent.step`、Policy、Executor 和 Reducer 等观测边界，不再拥有或
拼装完整 trace；Sink 和序列化失败不会改变业务执行结果。

每个组合完成的 Runtime bundle 只创建一个 `Tracer` 作为该调用链的 tracing owner。
Runtime 使用它建立根 Span，LLM、Retrieval 等深层组件不创建自己的 `Tracer`，而是
调用模块级 `start_span()` 加入当前 trace。不同 Runtime 或并发 Agent run 仍可拥有
彼此独立的 trace，并由 `ContextVar` 隔离。

真实 Runtime 通过 provider-independent 的 `TracingLLMClient` 装饰器，为 LLM Policy、
证据抽取和总结统一记录 `llm.call` Span。Span 输入记录实际发送的
`LLMRequest`，输出只记录进入业务层的 completion、provider 原始响应、模型和 token
usage；其状态只表示模型调用是否成功，不混入后续解析和业务校验结果。

调用方通过 `LLMRequest.metadata.trace` 提供 operation 等关联属性；证据抽取额外记录
文档和 chunk 位置。解析后的业务结果继续由 Step、Observation 和 ArtifactStore 持有，
可恢复的证据解析失败继续写入 `evidence_extraction_failures`，其他解析异常由既有外层
Span 记录。上述观测数据通过当前 trace 直接写入 Sink，不通过业务返回值逐层传递。

后续可以支持：

```text
trace replay
offline evaluation
policy comparison
reward modeling
LLM RL rollout/training data generation
```

后续 Policy 版本应使用相同 evaluation cases 和指标进行对比，为 reward
设计和 LLM RL 后训练提供可复现的行为基线。Reward、
自动评分和其他 evaluator 才能获得的信息应作为 rollout 完成后的训练标注保存，
不能泄漏到生成当前 action 的 policy 输入中。

## 阶段 7：官方来源后台摄取与本地检索

目标：建立一套独立于 News Agent loop 的后台摄取系统，持续保存经过来源审查的
官方与研究资料；Agent 可选择本地检索，后续再以 Tavily 补充覆盖缺口。

最终数据流：

```text
官方来源注册表
    -> RSS/Atom 与 Sitemap 增量发现
    -> robots 校验与受控页面获取
    -> HTML/PDF 解析
    -> SQLite 最新文档语料库
    -> 段落感知分块
    -> LanceDB BM25/vector/hybrid 索引
    -> Agent 本地检索
    -> Agent 按次选择 Local 或 Web route
```

边界与约束：

- 官方来源注册表是摄取范围的权威配置，记录来源身份、允许域名、RSS/Sitemap
  endpoint 和路径规则；“已审查来源”表示 provenance 可信，不代表单篇内容已经
  完成事实核验。
- RSS/Atom 负责近期更新发现，Sitemap 负责站点 URL 清单与历史回填；只获取二者
  明确发现且通过注册表范围校验的页面，不进行链接爬取。
- 获取页面前遵守并按 origin 缓存 `robots.txt`；禁止访问的页面记录状态，但不获取
  或索引正文。
- SQLite 是权威存储，仅保留每个文档的最新正文；不可用内容保留记录并标为 inactive，
  不进入检索结果。
- LanceDB 是可从 SQLite 幂等重建的派生索引，保存段落感知 chunk、全文索引和
  embedding；查询可按次选择 BM25、精确向量或 RRF 混合检索。当前语料规模先使用
  cosine flat vector search，达到需要 ANN 的规模后再增加 HNSW，不改变检索接口。
- 后台同步先由手动 CLI 触发，不引入 scheduler；后台模型使用独立的 corpus 状态，
  不复用 Agent rollout 内的 active/shelved/unusable 生命周期。
- 后台摄取流程复用现有 URL 工具和 `DocumentParser`，可独立于 Agent 运行。Tavily-only
  runtime 不依赖 corpus；local-only runtime 显式依赖已准备好的 corpus。后台失败不影响
  Tavily-only runtime。

实施拆分：

1. 已完成与现有代码相交的兼容性重构：解析器复用、URL 工具集中和文档校准。
2. 已新增 JSON 官方来源注册表与 `SQLiteCorpusStore`。
3. 已完成 RSS/Atom、Sitemap、robots 和增量同步服务，包括 discovery endpoint
   与内容页面条件请求、嵌套 Sitemap、来源范围校验以及 SQLite 写入编排。
4. 已新增段落感知分块和可从 SQLite 幂等重建的 LanceDB
   BM25/vector/hybrid 索引；未变化 chunk 可复用已有 embedding。
5. 已新增语料管理 CLI、多个已审查来源、CLI 端到端测试与运维文档；
   后续来源仍需逐个核实后加入。
6. 已将适配层接入真实 News Runtime，可启用 local-only、Web-only 或同时启用两路；
   同时启用时由 LLM Policy 为每次 Research 选择 route，当前不自动 fallback。

## 阶段 8：LLM Policy RL 后训练预留

目标：利用 agent rollout、结果评估和 reward 对 LLM Agent Policy 进行 RL 后训练，
优化的对象是生成 `AgentAction` 的 LLM 参数，而不是引入一个独立的传统
`RLPolicy` 来替换 LLM policy。

预留接口：

```text
RolloutRunner
RolloutStore
RewardModel
LLMPolicyTrainer
Evaluator
```

运行时继续使用统一的 `Policy` 接口和 `LLMAgentPolicy` 实现。训练侧消费完整、
可复现的 rollout，并在 episode 结束后附加 reward。核心数据流为：

```text
policy prompt/messages
    -> LLM completion
    -> parsed AgentAction
    -> observation
    -> next policy prompt/messages
    -> episode outcome
    -> reward
    -> update LLM parameters
```

Rollout 必须保存 LLM 实际接收的 prompt/messages、原始 completion、解析和校验
过程、action/observation、终止原因以及模型与配置版本，不能假设未来可以仅凭
`AgentState` 精确重建当时的训练样本。具体 RL 算法和训练框架在该阶段再选型，
当前不提前绑定 PPO、GRPO 或其他实现。

## 初始技术方向

第一版建议使用：

- Python
- Pydantic
- asyncio
- InMemory TraceSink
- SQLite 作为本地语料库的权威存储；Trace 等其他数据的持久化方案后续单独选择
- 自研轻量 `AgentRuntime`

外部 agent 框架暂不作为核心依赖。LangGraph、LangChain、LlamaIndex、Haystack 等后续可以作为 adapter 接入。


## 已知问题

### 文档获取与异常处理

- Document fetch 已对已知可恢复失败进行 action-local 有界重试；retrieval provider
  的短暂连接错误仍会使整次 Research 失败，也缺少按失败类别聚合的指标和持久化日志。
- PDF 目前仅支持通过 `pypdf` 提取文本层；扫描件 OCR、复杂版面、表格和公式恢复仍
  需要结合真实语料评估更强的解析方案。
- HTML 正文抽取仍是启发式的，缺少质量判定和低质量结果的回退策略。

### 证据提取

- 当前 evidence extraction 的输入预算估算较为粗糙，在不同模型和 provider 配置下
  可能不够准确。
- 分块提取结果尚未去重，重复 evidence 可能增加后续 synthesis 的上下文开销。

### Trace

- Trace 仍未持久化 artifact 内容；LLM prompt 和响应可随评测 trace JSONL 保存，但
  常规运行使用内存 Sink，进程结束后仍无法独立审计或完整回放。
- TraceSink 写入失败已经与业务异常隔离，但目前会被静默忽略，后续需要增加不会反向
  影响业务执行的诊断日志。
- Evaluation 当前仍从 `agent.step` Span 的输出还原 Action 和 Observation。二者已经
  存在于 `AgentState.action_history`，后续应以 State 作为业务事实来源，只从 Span
  获取阶段耗时、失败和 LLM usage 等观测数据，避免评估结果依赖 best-effort Trace。

### 本地语料检索

- BM25 与向量检索当前共用无重叠的简单段落分块；后续需要根据检索与端到端
  evaluation，在召回完整性、结果重复度和索引成本之间评估是否引入重叠、相邻
  chunk 扩展或更复杂的分块策略。
- 本地 corpus 与 Web route 可同时启用，但一次 `RESEARCH` 只执行一路且没有自动
  fallback。Route 由 LLM Policy 按次选择；Research 内对新旧 SearchResults 仍按 provider
  order 有界选取，后续可在该边界加入 reranker。

### 来源分类

- 来源类型注册表仍需要人工维护，可能漏掉新出现或低频的官方、研究、政府与新闻域名。
  未命中注册表且 provider 未提供有效类型的结果仍会标记为 `unknown`，需要继续通过
  evaluation 监测覆盖率和高频未知域名。
