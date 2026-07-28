"""Tests for runtime tracing and partial failure records."""

import asyncio
import json
from types import SimpleNamespace

import pytest

import scripts.evaluate_news_runtime as evaluation_script
from banso.apps.news_evaluation import NewsEvaluationCase, NewsEvaluationResult
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
from banso.tracing import (
    InMemoryTraceSink,
    SpanRecord,
    Tracer,
    get_current_span,
    start_span,
)


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


def _traced_runtime(**kwargs) -> tuple[AgentRuntime, InMemoryTraceSink]:
    sink = InMemoryTraceSink()
    return AgentRuntime(**kwargs, tracer=Tracer(sink)), sink


def _run_and_capture(
    runtime: AgentRuntime,
) -> RuntimeExecutionError:
    with pytest.raises(RuntimeExecutionError) as caught:
        asyncio.run(runtime.run(AgentState(query=UserQuery(text="test query"))))
    return caught.value


def _failed_span(spans: list[SpanRecord], name: str) -> SpanRecord:
    return next(
        span for span in spans if span.name == name and span.status == "error"
    )


def test_runtime_budget_exhaustion_does_not_mark_task_done() -> None:
    runtime, sink = _traced_runtime(
        policy=ContinuePolicy(),
        executor=StopExecutor(),
    )
    output = asyncio.run(
        runtime.run(
            AgentState(
                query=UserQuery(text="test query"),
                budget=ExecutionBudget(max_steps=1),
            )
        )
    )

    spans = sink.get_trace(output.trace_id)
    assert len([span for span in spans if span.name == "agent.step"]) == 1
    assert output.result.state.current_step == 1
    assert output.result.state.done is False
    assert _failed_span_or_none(spans, "agent.run") is None


def test_runtime_stop_marks_task_done_before_budget_exhaustion() -> None:
    runtime, sink = _traced_runtime(policy=StopPolicy(), executor=StopExecutor())
    output = asyncio.run(
        runtime.run(
            AgentState(
                query=UserQuery(text="test query"),
                budget=ExecutionBudget(max_steps=1),
            )
        )
    )

    assert len(sink.get_trace(output.trace_id)) == 5
    assert output.result.state.done is True
    assert output.result.state.last_action == AgentActionType.STOP


def test_runtime_joins_an_active_trace() -> None:
    runtime, sink = _traced_runtime(policy=StopPolicy(), executor=StopExecutor())

    async def run_nested():
        with runtime.tracer.start_span("request") as request_span:
            output = await runtime.run(
                AgentState(query=UserQuery(text="test query"))
            )
            return request_span, output

    request_span, output = asyncio.run(run_nested())
    spans = sink.get_trace(output.trace_id)
    run_span = next(span for span in spans if span.name == "agent.run")

    assert output.trace_id == request_span.trace_id
    assert run_span.parent_span_id == request_span.span_id


def test_runtime_records_policy_failure() -> None:
    runtime, sink = _traced_runtime(
        policy=RaisingPolicy(),
        executor=StopExecutor(),
    )
    error = _run_and_capture(runtime)
    spans = sink.get_trace(error.trace_id)

    failure = _failed_span(spans, "agent.policy.select")
    assert isinstance(error.original_error, ValueError)
    assert failure.error is not None
    assert failure.error.error_type == "ValueError"
    assert failure.error.message == "policy failed"
    assert failure.attributes["step_index"] == 0
    failed_step = _failed_span(spans, "agent.step")
    assert failed_step.input["state"]["current_step"] == 0
    assert failed_step.output is None


def test_runtime_records_executor_failure() -> None:
    runtime, sink = _traced_runtime(
        policy=StopPolicy(),
        executor=RaisingExecutor(),
    )
    error = _run_and_capture(runtime)
    spans = sink.get_trace(error.trace_id)

    failure = _failed_span(spans, "agent.action.execute")
    assert isinstance(error.original_error, OSError)
    assert failure.error is not None
    assert failure.error.message == "executor failed"
    failed_step = _failed_span(spans, "agent.step")
    assert failed_step.output["action"] == {
        "type": "stop",
        "params": {},
        "rationale": None,
    }
    assert "observation" not in failed_step.output


def test_runtime_records_reducer_failure_and_serializes_spans() -> None:
    runtime, sink = _traced_runtime(
        policy=StopPolicy(),
        executor=StopExecutor(),
        reducer=RaisingReducer(),
    )
    error = _run_and_capture(runtime)
    spans = sink.get_trace(error.trace_id)

    failure = _failed_span(spans, "agent.state.reduce")
    assert failure.error is not None
    assert failure.error.error_type == "RuntimeError"
    failed_step = _failed_span(spans, "agent.step")
    assert failed_step.output["observation"] == {"data": {}}
    restored = [
        SpanRecord.model_validate_json(span.model_dump_json()) for span in spans
    ]
    assert restored == spans


def test_trace_sink_failure_does_not_fail_runtime() -> None:
    class RaisingSink:
        def write(self, span: SpanRecord) -> None:
            raise ValueError(f"cannot write {span.name}")

    runtime = AgentRuntime(
        policy=StopPolicy(),
        executor=StopExecutor(),
        tracer=Tracer(RaisingSink()),
    )

    output = asyncio.run(
        runtime.run(AgentState(query=UserQuery(text="test query")))
    )

    assert output.result.state.done is True
    assert output.trace_id


def test_context_propagates_to_concurrent_tasks_and_resets() -> None:
    sink = InMemoryTraceSink()
    tracer = Tracer(sink)

    async def run_children() -> str:
        with tracer.start_span("root") as root:
            async def child(name: str) -> None:
                with start_span(name):
                    await asyncio.sleep(0)

            await asyncio.gather(child("child.one"), child("child.two"))
            assert get_current_span() is root
            return root.trace_id

    trace_id = asyncio.run(run_children())

    assert get_current_span() is None
    spans = sink.get_trace(trace_id)
    root = next(span for span in spans if span.name == "root")
    children = [span for span in spans if span.name.startswith("child.")]
    assert len(children) == 2
    assert all(span.parent_span_id == root.span_id for span in children)

    with tracer.start_span("next") as next_root:
        next_trace_id = next_root.trace_id
    assert next_trace_id != trace_id
    assert sink.get_trace(next_trace_id)[0].parent_span_id is None


def test_trace_serialization_failure_does_not_escape() -> None:
    sink = InMemoryTraceSink()
    tracer = Tracer(sink)
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic

    with tracer.start_span("cyclic", input=cyclic) as span:
        trace_id = span.trace_id

    record = sink.get_trace(trace_id)[0]
    assert record.status == "ok"
    assert record.input["serialization_error"] == "ValueError"


def test_evaluation_keeps_runtime_failure_trace_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, sink = _traced_runtime(
        policy=RaisingPolicy(),
        executor=StopExecutor(),
    )
    bundle = SimpleNamespace(
        runtime=runtime,
        store=InMemoryArtifactStore(),
        trace_sink=sink,
    )
    monkeypatch.setattr(
        evaluation_script,
        "build_real_news_runtime",
        lambda: bundle,
    )

    result, spans = asyncio.run(
        evaluation_script.run_case(
            NewsEvaluationCase(
                id="failure-case",
                category="runtime",
                query="test query",
            ),
            max_documents_to_read=1,
            max_active_documents=None,
        )
    )

    assert result.error_type == "ValueError"
    assert result.error_message == "policy failed"
    assert result.trace_id is not None
    assert _failed_span(spans, "agent.policy.select").error is not None


def test_evaluation_writes_in_memory_trace_jsonl(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = NewsEvaluationCase(
        id="saved-case",
        category="runtime",
        query="test query",
    )
    sink = InMemoryTraceSink()
    tracer = Tracer(sink)
    with tracer.start_span("agent.run") as span:
        trace_id = span.trace_id
    spans = sink.get_trace(trace_id)

    async def fake_run_case(
        case,
        *,
        max_documents_to_read,
        max_active_documents,
    ):
        del max_documents_to_read, max_active_documents
        return (
            NewsEvaluationResult(
                case_id=case.id,
                category=case.category,
                query=case.query,
                trace_id=trace_id,
            ),
            spans,
        )

    monkeypatch.setattr(
        evaluation_script,
        "load_evaluation_cases",
        lambda path: [case],
    )
    monkeypatch.setattr(evaluation_script, "run_case", fake_run_case)
    output_path = tmp_path / "results.jsonl"

    asyncio.run(
        evaluation_script.main(
            SimpleNamespace(
                cases=tmp_path / "cases.jsonl",
                output=output_path,
                limit=None,
                max_documents_to_read=1,
                max_active_documents=None,
            )
        )
    )

    trace_path = tmp_path / "results.traces.jsonl"
    saved_trace = json.loads(trace_path.read_text())
    assert saved_trace["trace_id"] == trace_id
    assert saved_trace["evaluation_case_id"] == case.id
    assert [saved["name"] for saved in saved_trace["spans"]] == ["agent.run"]


def _failed_span_or_none(
    spans: list[SpanRecord],
    name: str,
) -> SpanRecord | None:
    return next(
        (
            span
            for span in spans
            if span.name == name and span.status == "error"
        ),
        None,
    )
