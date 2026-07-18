# News Agent 初步计划

## 背景

目标是设计并逐步实现一个“新闻搜索 + 信息筛选 + 总结”的 agent 系统。当前已完成
固定流程 MVP、LLM Policy 所需的 State、Artifact 与 Policy View 基础，以及根据这些
输入动态选择 Action 的最小 `LLMNewsPolicy`。下一步将 LLM Policy 接入真实运行入口，
并补充独立的 LLM tracing。

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
- 已定义核心 runtime、state、action、policy、executor、reducer、result 等基础模块。
- 已实现最小 `AgentRuntime` 主循环，支持 policy 决策、executor 执行、reducer 更新状态和 trace 收集。
- 已实现新闻场景的固定流程 policy：搜索、读取文档、抽取 evidence、生成总结。
- 已实现最小 `LLMNewsPolicy`，由 LLM 根据有界 Policy View、可用 action 和剩余预算
  选择结构化 `AgentAction`，并校验动作参数、搜索预算、重复 query 和资源前置条件。
- `LLMNewsPolicy` 的非法输出和已知 LLM 调用失败会作为带 reason 的 policy error
  交由 Runtime 保存 partial trace；第一版不重试或自动回退。
- 尚未将 `LLMNewsPolicy` 接入真实新闻运行入口，也尚未记录 prompt、原始响应和
  token usage；这是下一步工作。
- 已实现可替换的 retrieval、document reader、evidence extractor、synthesizer 和 LLM client 接口及部分实现。
- 已接入 Tavily retrieval、HTTP document reader、OpenAI SDK LLM client、LLM evidence extractor 和 LLM synthesizer。
- 已实现超长文档的分块 evidence extraction，并隔离单篇文档的 LLM 提取失败。
- 已实现 retrieval filter、仅补充元数据而不做准入的 search result source
  classifier，以及基础 artifact store。
- 已定义 `AgentTrace` 和 `TraceStep` 数据结构，用于记录运行轨迹。
- `AgentState` 已保存有界的 Action/Observation 历史、artifact ID、最终答案和
  citations；完整 artifact 继续由 `ArtifactStore` 作为权威数据源保存。
- 内存 ArtifactStore 已保证同 ID 不可覆盖，并在写入、读取和列举时提供隔离快照。
- 已实现新闻专用的 `NewsPolicyStateViewBuilder`，按 State 中的 ID 顺序构造有界的
  SearchResult、Document 和 Evidence Policy View。
- TraceStep 已分别记录 Policy、Executor 和 Reducer 耗时；失败 Trace 记录失败阶段及
  该阶段耗时。
- 已补充覆盖核心 runtime、新闻执行器、retrieval、document reader、LLM 配置和 LLM 组件的测试。

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
DocumentReader
DocumentRanker
EvidenceExtractor
Synthesizer
TraceLogger
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
- `TraceLogger` 接口
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

## 阶段 5：LLM Agent Policy

目标：在保持 runtime、action space 和 executor 边界不变的前提下，让 LLM 根据
当前 state、已有 observation 和剩余预算动态选择下一步 action。

第一版使用受约束的已有动作空间：

```text
PLAN_SEARCH
SEARCH
READ_DOCUMENT
EXTRACT_EVIDENCE
SYNTHESIZE
STOP
```

核心要求：

- 输出可校验的结构化 `AgentAction`，不允许生成任意工具调用。
- `LLMNewsPolicy` 内部使用 `NewsPolicyStateViewBuilder`，根据 State 和 ArtifactStore
  构造模型可见输入；通用 `Policy` 接口继续只返回 `AgentAction`。
- 向模型提供当前 state、已完成步骤、有界 artifact view、可用 action 和剩余预算，
  不在 State 中重复保存 artifact summary。
- 对非法 action、无效参数、重复动作和 LLM 调用失败提供校验、重试或安全回退。
- 记录 action 选择所需的简短 decision metadata，保证行为可审计。
- 继续使用固定流程 policy 作为 baseline，而不是直接替换或删除。
- 通过最大步骤数、搜索数、文档数和 token/cost 预算约束 agent 行为。

当前已完成最小 Policy 本身及其确定性输出校验；应用入口接入、独立 LLM tracing、
token/cost 预算和基于评估结果的重试策略仍属于后续工作。

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

当前 Runtime Trace 负责 State、Action、Observation、阶段耗时和失败信息。LLM 实际
接收的 messages、原始响应、token usage 和解析过程属于独立的 LLM tracing 能力，
不通过通用 Policy 返回模型暴露给规则 Policy。

后续可以支持：

```text
trace replay
offline evaluation
policy comparison
reward modeling
LLM RL rollout/training data generation
```

LLM Agent Policy 与固定流程 policy 应使用相同 evaluation cases 和指标进行
对比，为后续 reward 设计和 LLM RL 后训练提供可复现的行为基线。Reward、
自动评分和其他 evaluator 才能获得的信息应作为 rollout 完成后的训练标注保存，
不能泄漏到生成当前 action 的 policy 输入中。

## 阶段 7：LLM Policy RL 后训练预留

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
- JSONL trace
- SQLite 或 Postgres 作为后续持久化选项
- 自研轻量 `AgentRuntime`

外部 agent 框架暂不作为核心依赖。LangGraph、LangChain、LlamaIndex、Haystack 等后续可以作为 adapter 接入。


## 已知问题

### 文档读取与异常处理

- HTTP 429、可恢复的 5xx、超时和连接错误尚未实现有上限的重试，也缺少按
  失败类别聚合的指标和持久化日志。
- HTML 正文抽取仍是启发式的，缺少质量判定和低质量结果的回退策略。

### 证据提取

- 当前 evidence extraction 的输入预算估算较为粗糙，在不同模型和 provider 配置下
  可能不够准确。
- 分块提取结果尚未去重，重复 evidence 可能增加后续 synthesis 的上下文开销。

### Trace

- Trace 仍未持久化 artifact 内容，也尚未记录 LLM 实际收到的 messages 和原始响应，
  进程结束后无法独立审计或完整回放。

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

### 来源分类

- 来源类型注册表仍需要人工维护，可能漏掉新出现或低频的官方、研究、政府与新闻域名。
  未命中注册表且 provider 未提供有效类型的结果仍会标记为 `unknown`，需要继续通过
  evaluation 监测覆盖率和高频未知域名。
