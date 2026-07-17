"""Tests for runtime partial traces."""

import asyncio
from types import SimpleNamespace

import pytest

import scripts.evaluate_news_runtime as evaluation_script
from banso.apps.news_evaluation import NewsEvaluationCase
from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    AgentAction,
    AgentActionType,
    AgentRuntime,
    AgentState,
    ExecutionBudget,
    Observation,
    RuntimeExecutionError,
    UserQuery,
)
from banso.tracing import AgentTrace


class StopPolicy:
    async def select_action(self, state: AgentState) -> AgentAction:
        return AgentAction(type=AgentActionType.STOP)


class RaisingPolicy:
    async def select_action(self, state: AgentState) -> AgentAction:
        raise ValueError("policy failed")


class ContinuePolicy:
    async def select_action(self, state: AgentState) -> AgentAction:
        return AgentAction(type=AgentActionType.SEARCH)


class StopExecutor:
    async def execute(
        self,
        action: AgentAction,
        state: AgentState,
    ) -> Observation:
        return Observation()


class RaisingExecutor:
    async def execute(
        self,
        action: AgentAction,
        state: AgentState,
    ) -> Observation:
        raise OSError("executor failed")


class RaisingReducer:
    def apply(
        self,
        state: AgentState,
        action: AgentAction,
        observation: Observation,
    ) -> AgentState:
        raise RuntimeError("reducer failed")


def _run_and_capture(runtime: AgentRuntime) -> RuntimeExecutionError:
    with pytest.raises(RuntimeExecutionError) as caught:
        asyncio.run(runtime.run(AgentState(query=UserQuery(text="test query"))))
    return caught.value


def test_runtime_budget_exhaustion_does_not_mark_task_done() -> None:
    output = asyncio.run(
        AgentRuntime(policy=ContinuePolicy(), executor=StopExecutor()).run(
            AgentState(
                query=UserQuery(text="test query"),
                budget=ExecutionBudget(max_steps=1),
            )
        )
    )

    assert len(output.trace.steps) == 1
    assert output.result.state.current_step == 1
    assert output.result.state.done is False
    assert output.trace.status == "completed"


def test_runtime_stop_marks_task_done_before_budget_exhaustion() -> None:
    output = asyncio.run(
        AgentRuntime(policy=StopPolicy(), executor=StopExecutor()).run(
            AgentState(
                query=UserQuery(text="test query"),
                budget=ExecutionBudget(max_steps=1),
            )
        )
    )

    assert len(output.trace.steps) == 1
    assert output.result.state.done is True
    assert output.result.state.last_action == AgentActionType.STOP


def test_runtime_records_policy_failure() -> None:
    error = _run_and_capture(
        AgentRuntime(policy=RaisingPolicy(), executor=StopExecutor())
    )

    failure = error.trace.failure
    assert error.trace.status == "failed"
    assert error.trace.final_result is None
    assert error.trace.steps == []
    assert failure is not None
    assert failure.phase == "policy"
    assert failure.step_index == 0
    assert failure.action is None
    assert failure.observation is None
    assert failure.error_type == "ValueError"
    assert failure.message == "policy failed"
    assert failure.state.current_step == 0
    assert failure.duration_seconds is not None
    assert failure.duration_seconds >= 0


def test_runtime_records_executor_failure() -> None:
    error = _run_and_capture(
        AgentRuntime(policy=StopPolicy(), executor=RaisingExecutor())
    )

    failure = error.trace.failure
    assert isinstance(error.original_error, OSError)
    assert failure is not None
    assert failure.phase == "executor"
    assert failure.action == AgentAction(type=AgentActionType.STOP)
    assert failure.observation is None
    assert error.trace.steps == []


def test_runtime_records_trace_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_to_build_trace_step(**kwargs):
        raise ValueError("trace failed")

    monkeypatch.setattr("banso.core.runtime.TraceStep", fail_to_build_trace_step)
    error = _run_and_capture(AgentRuntime(policy=StopPolicy(), executor=StopExecutor()))

    failure = error.trace.failure
    assert failure is not None
    assert failure.phase == "trace"
    assert failure.action == AgentAction(type=AgentActionType.STOP)
    assert failure.observation == Observation()
    assert error.trace.steps == []


def test_runtime_records_reducer_failure_and_serializes_trace() -> None:
    error = _run_and_capture(
        AgentRuntime(
            policy=StopPolicy(),
            executor=StopExecutor(),
            reducer=RaisingReducer(),
        )
    )

    failure = error.trace.failure
    assert failure is not None
    assert failure.phase == "reducer"
    assert failure.observation == Observation()
    assert len(error.trace.steps) == 1

    restored = AgentTrace.model_validate_json(error.trace.model_dump_json())
    assert restored.status == "failed"
    assert restored.failure == failure


def test_evaluation_keeps_runtime_failure_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = SimpleNamespace(
        runtime=AgentRuntime(policy=RaisingPolicy(), executor=StopExecutor()),
        store=InMemoryArtifactStore(),
    )
    monkeypatch.setattr(
        evaluation_script,
        "build_real_news_runtime",
        lambda: bundle,
    )

    result, trace = asyncio.run(
        evaluation_script.run_case(
            NewsEvaluationCase(
                id="failure-case",
                category="runtime",
                query="test query",
            ),
            max_documents_to_read=1,
        )
    )

    assert result.error_type == "ValueError"
    assert result.error_message == "policy failed"
    assert trace is not None
    assert trace.status == "failed"
    assert trace.failure is not None
    assert trace.failure.phase == "policy"
