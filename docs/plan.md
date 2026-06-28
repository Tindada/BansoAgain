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

## 下一步

准备开发环境：

1. 确认 Python 版本与包管理工具。
2. 初始化项目结构。
3. 配置基础依赖。
4. 建立代码格式化、类型检查和测试工具。
5. 创建最小包结构，为核心接口设计做准备。
