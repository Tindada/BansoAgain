# Agent State 设计讨论摘要

> 临时文档，用于记录当前讨论结论，后续会随 Action 空间和 Policy 设计继续调整。

## 目标与边界

当前暂不实现 Agent Policy，先补充未来 Policy 做动作选择所需的 State 结构。

基本关系：

- `Action` 定义 Policy 可以做什么以及可传递的参数。
- `Executor` 定义动作的实际执行语义。
- `State` 定义 Policy 做决定时可以看到什么。
- `Reducer` 将 Observation 中仍有决策价值的信息更新到 State。

State 是否完整需要结合 Action 空间判断。对于新闻、模型发布和排行榜更新等业务，Agent 更适合负责有语义价值的搜索与信息选择决策，底层抓取、去重、解析、重试和预算强制约束仍由确定性组件负责。

暂时不预先定义复杂的事件分类、搜索覆盖维度或“信息已经足够全面”等结论。LLM 可以基于搜索历史、产物内容和预算自行判断。

## State 的核心内容

第一版 State 保持轻量，主要增加两类信息：

1. 当前运行中各步 Action 及其 Observation。
2. 已产生 artifact 的 ID，包括 search result、document 和 evidence。

可参考以下结构：

```python
class AgentState(BaseModel):
    query: UserQuery
    current_step: int
    budget: ExecutionBudget
    search_plan: SearchPlan | None

    action_history: list[ActionHistoryEntry]

    search_result_ids: list[str]
    document_ids: list[str]
    evidence_ids: list[str]

    final_answer: str | None
    citations: list[str]
    last_action: AgentActionType | None
    done: bool
    termination_reason: str | None
```

周期任务相关的历史基线暂不作为当前设计重点。

## Action 历史与 Observation

不再额外维护一套 `ActionResultSummary`，直接将有界的 Observation 作为 Action 执行结果保存：

```python
class ActionHistoryEntry(BaseModel):
    step_index: int
    action_type: AgentActionType
    params: dict[str, Any]
    observation: Observation
```

过去 Action 的完整 rationale 不必进入 State，可以保留在 Trace 中。

Observation 应当是可直接提供给 Policy 的结构化执行结果，而不是底层工具的原始响应。建议增加统一状态：

```python
class Observation(BaseModel):
    data: dict[str, JsonValue] = Field(default_factory=dict)
```

Observation 可以包含 artifact ID、数量、过滤报告和失败信息，但不应包含完整网页正文、完整 provider 响应、LLM 原始 completion 或无界 metadata。

批量处理 Action 应在 `data` 中使用带计数单位的字段，避免通用的 `success_count`、
`failure_count` 和 `output_count` 产生歧义。例如文档读取记录
`successfully_read_document_count` 和 `failed_document_count`；Evidence 提取记录
`successful_document_count`、`failed_document_count` 和 `evidence_count`。具体失败
原因继续使用 Action 对应的结构化字段，例如 `document_read_failures` 或
`evidence_extraction_failures`。Policy 可以结合数量和失败原因决定是否重试，不额外
维护一套重复的通用状态和错误模型。

如果某个失败需要由 Policy 决定是否重试，应将其表示为 Observation；只有不可恢复的运行时错误才直接终止运行。

## Artifact 与 Policy View

State 中仍只保存 artifact ID，不复制保存一套 Summary 字段。完整 artifact 由 `ArtifactStore` 管理。

每种 artifact 提供一个纯转换方法，生成有界的 Policy View：

```python
class Document(BaseModel):
    ...

    def to_policy_view(self) -> dict[str, Any]:
        ...
```

`to_policy_view()` 应满足：

- 不调用 LLM 或网络；
- 不修改 artifact；
- 输出有界、结构化且可序列化；
- 相同 artifact 和配置产生相同结果。

Search Result 的 View 主要包含标题、URL、snippet、来源和发布时间；Document 的 View 不包含完整正文，只包含元数据和有界预览或已生成摘要；Evidence 的 View 主要包含 claim、来源、时间和 confidence，不包含较长的 supporting text。

如果需要真正的文档语义摘要，应在执行阶段生成并保存，不能在 `to_policy_view()` 中临时调用 LLM。

## Policy 输入构建

在调用 Policy 前，由独立的构造层根据 State 中的 ID 加载 artifact 并生成 View：

```text
AgentState
    + ArtifactStore
    + PolicyStateViewBuilder
            ↓
      PolicyStateView
            ↓
          Policy
```

因此，Policy 输入不是只凭 State 就能恢复，而是由以下内容确定：

```text
AgentState + ArtifactStore + PolicyStateViewBuilder 版本
```

需要保证 artifact 在写入后不可被同 ID 覆盖修改、ID 缺失不会被静默忽略、View 构造和截断规则稳定，并按照 State 中的 ID 顺序生成 View。

为了支持精确回放和未来 RL 训练，Trace/Rollout 仍应保存模型当时实际收到的 prompt/messages，不能只依赖之后重新构建 Policy View。

## 当前倾向

- State 主要保存动作历史、Observation、artifact ID 和基础运行状态。
- ArtifactStore 保存完整产物并作为权威数据源。
- PolicyStateViewBuilder 在决策前按需物化 Policy 可见内容。
- 不在 State 中重复维护 SearchResultSummary、DocumentSummary 和 EvidenceSummary。
- 暂时保留通用 `Observation.data`，等 Action 空间稳定后再考虑按 Action 类型化。
