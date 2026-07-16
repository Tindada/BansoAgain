# News Agent 初步计划

## 背景

目标是设计并逐步实现一个“新闻搜索 + 信息筛选 + 总结”的 agent 系统。当前阶段先做系统设计和环境准备，不直接进入完整实现。

系统需要长期支持：

- 可替换 LLM
- 可替换搜索与 retrieval
- 可插入 RL policy
- 完整执行轨迹记录
- 可评估、可回放、可迭代优化

核心原则：

> 核心 runtime 自研，外部能力插件化接入。

## 当前状态

已完成：

- 已初始化 Python 项目结构，并使用 `uv` 管理依赖。
- 已定义核心 runtime、state、action、policy、executor、reducer、result 等基础模块。
- 已实现最小 `AgentRuntime` 主循环，支持 policy 决策、executor 执行、reducer 更新状态和 trace 收集。
- 已实现新闻场景的固定流程 policy：搜索、读取文档、抽取 evidence、生成总结。
- 已实现可替换的 retrieval、document reader、evidence extractor、synthesizer 和 LLM client 接口及部分实现。
- 已接入 Tavily retrieval、HTTP document reader、OpenAI SDK LLM client、LLM evidence extractor 和 LLM synthesizer。
- 已实现超长文档的分块 evidence extraction，并隔离单篇文档的 LLM 提取失败。
- 已实现 retrieval filter、search result evaluator 和基础 artifact store。
- 已定义 `AgentTrace` 和 `TraceStep` 数据结构，用于记录运行轨迹。
- 已补充覆盖核心 runtime、新闻执行器、retrieval、document reader、LLM 配置和 LLM 组件的测试。

## 阶段 1：系统架构设计

目标：明确整个 agent 系统的边界、模块和数据流。

产出：

- 系统整体架构图
- 核心模块划分
- Agent 执行流程
- 未来 RL 扩展点说明

重点模块：

```text
AgentRuntime
Policy
AgentState
AgentAction
RetrievalProvider
DocumentReader
DocumentRanker
EvidenceExtractor
Synthesizer
TraceLogger
RewardModel
```

## 阶段 2：核心接口设计

目标：先定义稳定接口，不绑定具体实现。

产出：

- `AgentState` 数据结构
- `AgentAction` 动作空间
- `Policy` 接口
- `LLMProvider` 接口
- `RetrievalProvider` 接口
- `TraceLogger` 接口
- `RewardModel` 预留接口

设计原则：

```text
可替换 LLM
可替换搜索/检索
可替换 policy
可记录完整 trace
可支持未来 RL
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
trace logger 记录步骤
循环直到 STOP
```

第一版不追求复杂智能，先保证主循环清晰、可测试、可扩展。

## 阶段 4：新闻搜索链路 MVP

目标：实现最小新闻搜索与总结流程。

包含：

- Query 分析
- 生成搜索 query
- 调用搜索 provider
- 读取文档
- 筛选相关文档
- 抽取 evidence
- 多源总结
- 输出最终新闻摘要

## 阶段 5：Trace 与评估系统

目标：为后续优化和 RL 做准备。

需要记录：

```text
用户输入
每一步 state
每一步 action
工具调用输入
工具调用输出
中间判断
最终答案
用户反馈
自动评分
```

后续可以支持：

```text
trace replay
offline evaluation
policy comparison
reward modeling
RL training data generation
```

## 阶段 6：RL 扩展预留

目标：让系统可以从 LLM-driven agent 平滑扩展到 RL-driven agent。

预留接口：

```text
RLPolicy
RewardModel
ReplayBuffer
Environment
Evaluator
```

核心抽象保持为：

```text
state -> action -> observation -> next_state -> reward
```

## 初始技术方向

第一版建议使用：

- Python
- Pydantic
- asyncio
- JSONL trace
- SQLite 或 Postgres 作为后续持久化选项
- 自研轻量 `AgentRuntime`

外部 agent 框架暂不作为核心依赖。LangGraph、LangChain、LlamaIndex、Haystack 等后续可以作为 adapter 接入。


## 已知问题

### 文档读取与异常处理

- 当前文档读取流程仅隔离 HTTP 401、403 和 404，单篇读取失败不会终止整批新闻处理。
- 部分页面存在正文抽取不完整的问题。真实运行中曾出现原始 HTML 约 22 万字符，
  最终仅抽取出 75 字正文的情况，导致可用于 evidence extraction 的内容不足。

后续按优先级补充：

- HTTP 429：遵循 `Retry-After`，并使用有上限的指数退避。
- HTTP 5xx：对可恢复状态进行有限次数重试。
- 请求超时和连接错误：有限次数重试，最终失败后记录并跳过。
- 统一失败分类、指标和日志，便于区分来源限制与基础设施故障。

### 证据提取

- 当前 evidence extraction 的输入预算估算较为粗糙，在不同模型和 provider 配置下
  可能不够准确。
- 分块提取结果尚未去重，重复 evidence 可能增加后续 synthesis 的上下文开销。

### Trace

- Trace 当前主要记录 artifact ID，没有持久化文档正文及其提取元数据。进程结束
  后，无法仅依赖 Trace 检查文档的正文长度和提取策略，也无法完整审计 artifact
  内容；运行在异常中断时不会保留 partial trace。

### 检索规划

- LLM Search Planner 在处理包含 `latest`、`recent`、`last N days` 等相对时间
  约束的查询时，可能在用户未指定具体年份的情况下生成带有过时年份的检索词。
  真实评估中曾将“最近 30 天”的模型更新查询规划为包含 `2023 October` 的
  检索词，导致计划与原始时间范围不一致，并可能引入过时的搜索结果。
- `PlannedSearch.intent` 当前仅保存在计划、Action 参数和 Trace 中，不会影响
  实际发送给 retrieval provider 的检索请求。

### 检索执行

- 不同 Search 返回的结果仅在单次 Search 内去重，尚未跨整个检索计划进行全局
  去重。相同 URL 出现在多个查询结果中时，可能被重复保存和读取。

### 来源评估

- 来源识别覆盖不足，部分可信的官方或专业来源可能被识别为 `unknown_source`。
  真实评估中曾出现 Google Cloud、Meta AI、Stanford HAI 和欧盟机构等来源未被
  正确识别的情况。
- 来源评分尚未经过充分校准。真实评估中的搜索结果拒绝比例较高，可能将有效
  搜索结果与低质量来源一并过滤。
