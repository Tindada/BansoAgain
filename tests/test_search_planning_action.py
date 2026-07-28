"""Tests for search planning as an independently executable action."""

import asyncio
from datetime import datetime, timezone

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    AgentAction,
    AgentActionType,
    AgentRuntime,
    AgentState,
    DefaultStateReducer,
    ExecutionBudget,
    PlannedSearch,
    SearchPlan,
    UserQuery,
)
from banso.core.observation import (
    FinishObservation,
    PlanSearchObservation,
    RetrievalFilterReport,
    SearchObservation,
    SearchResultMergeReport,
    SourceClassificationReport,
)
from banso.core.state import (
    ActionHistoryEntry,
    DocumentState,
    ExtractProgress,
    Failure,
    SearchResultState,
)
from banso.documents import FakeDocumentFetcher, FakeEvidenceExtractor
from banso.executors import NewsActionExecutor
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import (
    FakeRetrievalProvider,
    OriginalQueryPlanner,
    SearchPlanningRequest,
)
from banso.synthesis import FakeSynthesizer
from banso.tracing import InMemoryTraceSink, SpanRecord, Tracer

REFERENCE_TIME = datetime(2026, 7, 24, 8, 30, tzinfo=timezone.utc)


def _search_observation(
    *,
    result_ids: list[str],
    index_updates: dict[str, str],
) -> SearchObservation:
    return SearchObservation(
        search_queries=["latest AI news"],
        search_result_ids=result_ids,
        search_result_index_updates=index_updates,
        search_result_merge_report=SearchResultMergeReport(
            candidate_count=len(result_ids),
            new_result_count=len(index_updates),
            reused_result_count=len(result_ids) - len(index_updates),
        ),
        retrieval_filter_report=RetrievalFilterReport(
            input_count=len(result_ids),
            output_count=len(result_ids),
        ),
        source_classification_report=SourceClassificationReport(
            input_count=len(result_ids),
            recognized_count=0,
            unknown_count=len(result_ids),
        ),
    )


class RecordingSearchQueryPlanner:
    def __init__(self, plan: SearchPlan) -> None:
        self.search_plan = plan
        self.requests: list[SearchPlanningRequest] = []

    async def plan(self, request: SearchPlanningRequest) -> SearchPlan:
        self.requests.append(request)
        return self.search_plan


def _news_executor(
    search_query_planner: RecordingSearchQueryPlanner,
) -> NewsActionExecutor:
    return NewsActionExecutor(
        store=InMemoryArtifactStore(),
        retrieval_provider=FakeRetrievalProvider(),
        document_fetcher=FakeDocumentFetcher(),
        evidence_extractor=FakeEvidenceExtractor(),
        synthesizer=FakeSynthesizer(),
        search_query_planner=search_query_planner,
    )


def test_plan_search_action_and_models_are_serializable() -> None:
    action = AgentAction(type=AgentActionType.PLAN_SEARCH)
    plan = SearchPlan(searches=[PlannedSearch(query="latest AI news")])

    assert action.model_dump(mode="json")["type"] == "plan_search"
    assert plan.model_dump(mode="json") == {
        "searches": [{"query": "latest AI news", "intent": "general"}]
    }


def test_original_query_planner_respects_search_budget() -> None:
    planner = OriginalQueryPlanner()
    query = UserQuery(text="latest AI news", language="en")

    plan = asyncio.run(
        planner.plan(
            SearchPlanningRequest(
                query=query,
                reference_time=REFERENCE_TIME,
                max_searches=1,
            )
        )
    )
    empty_plan = asyncio.run(
        planner.plan(
            SearchPlanningRequest(
                query=query,
                reference_time=REFERENCE_TIME,
                max_searches=0,
            )
        )
    )

    assert plan == SearchPlan(
        searches=[PlannedSearch(query="latest AI news", intent="general")]
    )
    assert empty_plan == SearchPlan()


def test_news_executor_executes_plan_search() -> None:
    plan = SearchPlan(
        searches=[PlannedSearch(query="AI release", intent="official")]
    )
    planner = RecordingSearchQueryPlanner(plan)
    executor = _news_executor(planner)
    state = AgentState(
        query=UserQuery(text="latest AI news", region="US"),
        reference_time=REFERENCE_TIME,
        budget=ExecutionBudget(max_searches=2),
    )

    observation = asyncio.run(
        executor.execute(
            AgentAction(type=AgentActionType.PLAN_SEARCH),
            state,
        )
    )

    assert planner.requests == [
        SearchPlanningRequest(
            query=state.query,
            reference_time=REFERENCE_TIME,
            max_searches=2,
        )
    ]
    assert observation == PlanSearchObservation(search_plan=plan)


def test_reducer_writes_search_plan_without_mutating_input_state() -> None:
    state = AgentState(query=UserQuery(text="latest AI news"))
    action = AgentAction(type=AgentActionType.PLAN_SEARCH)
    observation = PlanSearchObservation(
        search_plan=SearchPlan(
            searches=[PlannedSearch(query="AI release", intent="official")]
        )
    )

    next_state = DefaultStateReducer().apply(state, action, observation)

    assert state.search_plan is None
    assert state.current_step == 0
    assert next_state.search_plan == SearchPlan(
        searches=[PlannedSearch(query="AI release", intent="official")]
    )
    assert next_state.last_action == AgentActionType.PLAN_SEARCH
    assert next_state.current_step == 1
    assert state.action_history == []
    assert next_state.action_history == [
        ActionHistoryEntry(
            step_index=0,
            action=action,
            observation=observation,
        )
    ]


def test_reducer_stores_independent_action_history_snapshot() -> None:
    state = AgentState(query=UserQuery(text="latest AI news"))
    action = AgentAction(
        type=AgentActionType.SEARCH,
        params={"filters": {"domains": ["example.com"]}},
    )
    observation = _search_observation(
        result_ids=["result-1"],
        index_updates={"https://example.com/result-1": "result-1"},
    )

    next_state = DefaultStateReducer().apply(state, action, observation)
    action.params["filters"]["domains"].append("other.example")
    observation.search_result_ids.append("result-2")
    observation.search_result_index_updates[
        "https://example.com/result-2"
    ] = "result-2"

    history_entry = next_state.action_history[0]
    assert history_entry.action.params == {
        "filters": {"domains": ["example.com"]},
    }
    assert isinstance(history_entry.observation, SearchObservation)
    assert history_entry.observation.search_result_ids == ["result-1"]
    assert history_entry.observation.search_result_index_updates == {
        "https://example.com/result-1": "result-1"
    }


def test_reducer_merges_search_results_without_duplicates() -> None:
    state = AgentState(
        query=UserQuery(text="latest AI news"),
        search_results={"result-1": SearchResultState()},
        search_result_index={"https://example.com/first": "result-1"},
    )
    observation = _search_observation(
        result_ids=["result-1", "result-2"],
        index_updates={"https://example.com/second": "result-2"},
    )

    next_state = DefaultStateReducer().apply(
        state,
        AgentAction(type=AgentActionType.SEARCH),
        observation,
    )

    assert list(next_state.search_results) == ["result-1", "result-2"]
    assert next_state.search_result_index == {
        "https://example.com/first": "result-1",
        "https://example.com/second": "result-2",
    }


@pytest.mark.parametrize("updated_result_id", ["result-1", "result-2"])
def test_reducer_rejects_existing_search_result_index_key(
    updated_result_id: str,
) -> None:
    state = AgentState(
        query=UserQuery(text="latest AI news"),
        search_results={"result-1": SearchResultState()},
        search_result_index={"https://example.com/news": "result-1"},
    )
    observed_ids = ["result-1"]
    if updated_result_id != "result-1":
        observed_ids.append(updated_result_id)
    observation = _search_observation(
        result_ids=observed_ids,
        index_updates={"https://example.com/news": updated_result_id},
    )

    with pytest.raises(ValueError, match="contains an existing URL"):
        DefaultStateReducer().apply(
            state,
            AgentAction(type=AgentActionType.SEARCH),
            observation,
        )


def test_reducer_writes_final_answer_without_mutating_input_state() -> None:
    state = AgentState(query=UserQuery(text="latest AI news"))
    action = AgentAction(type=AgentActionType.FINISH)
    observation = FinishObservation(
        final_answer="Synthesized answer.",
        citations=["https://example.com/source"],
    )

    next_state = DefaultStateReducer().apply(state, action, observation)

    assert state.final_answer is None
    assert state.citations == []
    assert next_state.final_answer == "Synthesized answer."
    assert next_state.citations == ["https://example.com/source"]
    assert next_state.done is True


def test_reducer_replaces_citations_with_new_final_answer() -> None:
    state = AgentState(
        query=UserQuery(text="latest AI news"),
        final_answer="Old answer.",
        citations=["https://example.com/old"],
    )
    action = AgentAction(type=AgentActionType.FINISH)
    observation = FinishObservation(
        final_answer="New answer.",
        citations=["https://example.com/new"],
    )

    next_state = DefaultStateReducer().apply(state, action, observation)

    assert state.final_answer == "Old answer."
    assert state.citations == ["https://example.com/old"]
    assert next_state.final_answer == "New answer."
    assert next_state.citations == ["https://example.com/new"]


def test_rule_policy_uses_resource_lifecycle_after_searching() -> None:
    policy = NewsRuleBasedPolicy()
    state = AgentState(
        query=UserQuery(text="latest AI news"),
        search_plan=SearchPlan(),
        search_results={"result-1": SearchResultState()},
        documents={"document-1": DocumentState()},
    )

    action = asyncio.run(policy.select_action(state))
    assert action.type == AgentActionType.FETCH_DOCUMENTS

    state.search_results["result-1"] = SearchResultState(
        attempt_count=1,
        document_id="document-1",
    )
    action = asyncio.run(policy.select_action(state))
    assert action.type == AgentActionType.EXTRACT_EVIDENCE

    state.documents["document-1"].extraction = ExtractProgress(attempt_count=1)
    state.documents["document-1"].evidence_ids = ["evidence-1"]
    state.documents["document-1"].lifecycle_status = "active"
    action = asyncio.run(policy.select_action(state))
    assert action.type == AgentActionType.FINISH


def test_rule_policy_retries_eligible_failure() -> None:
    state = AgentState(
        query=UserQuery(text="latest AI news"),
        search_plan=SearchPlan(),
        search_results={
            "result-1": SearchResultState(
                attempt_count=1,
                failure=Failure(reason="timeout", retryable=True),
            )
        },
    )

    action = asyncio.run(NewsRuleBasedPolicy().select_action(state))

    assert action.type == AgentActionType.FETCH_DOCUMENTS


@pytest.mark.parametrize(
    ("evidence_ids", "expected"),
    [
        (["evidence-1"], AgentActionType.FINISH),
        ([], AgentActionType.STOP),
    ],
)
def test_rule_policy_uses_last_step_to_end(
    evidence_ids: list[str],
    expected: AgentActionType,
) -> None:
    state = AgentState(
        query=UserQuery(text="latest AI news"),
        current_step=11,
        documents=(
            {
                "document-1": DocumentState(
                    evidence_ids=evidence_ids,
                    lifecycle_status="active",
                )
            }
            if evidence_ids
            else {}
        ),
    )

    action = asyncio.run(NewsRuleBasedPolicy().select_action(state))

    assert action.type == expected


async def _run_runtime_with_bounded_plan():
    plan = SearchPlan(
        searches=[
            PlannedSearch(query="general AI update"),
            PlannedSearch(query="official AI release", intent="official"),
            PlannedSearch(query="AI research paper", intent="research"),
        ]
    )
    planner = RecordingSearchQueryPlanner(plan)
    trace_sink = InMemoryTraceSink()
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=_news_executor(planner),
        tracer=Tracer(trace_sink),
    )

    output = await runtime.run(
        AgentState(
            query=UserQuery(text="latest AI news"),
            budget=ExecutionBudget(max_searches=2),
        )
    )
    return output, trace_sink.get_trace(output.trace_id)


def test_runtime_executes_bounded_search_plan_in_order() -> None:
    output, spans = asyncio.run(_run_runtime_with_bounded_plan())

    assert [entry.action.type for entry in output.result.state.action_history] == [
        AgentActionType.PLAN_SEARCH,
        AgentActionType.SEARCH,
        AgentActionType.SEARCH,
        AgentActionType.FETCH_DOCUMENTS,
        AgentActionType.EXTRACT_EVIDENCE,
        AgentActionType.FINISH,
    ]
    assert [
        entry.observation.search_queries[0]
        for entry in output.result.state.action_history
        if entry.action.type == AgentActionType.SEARCH
        and isinstance(entry.observation, SearchObservation)
    ] == [
        "general AI update",
        "official AI release",
    ]
    assert "search_queries" not in output.result.state.model_dump()
    assert [
        entry.step_index for entry in output.result.state.action_history
    ] == list(range(6))
    assert [
        entry.action.type for entry in output.result.state.action_history
    ] == [
        AgentActionType(span.output["action"]["type"])
        for span in spans
        if span.name == "agent.step" and span.status == "ok"
    ]
    assert (
        output.result.state.action_history[-1].action.type
        == AgentActionType.FINISH
    )
    assert output.result.state.current_step == len(
        output.result.state.action_history
    )
    step_spans = [
        span for span in spans if span.name == "agent.step" and span.status == "ok"
    ]
    assert [len(span.input["state"]["action_history"]) for span in step_spans] == list(
        range(6)
    )
    assert step_spans[0].output["observation"]["search_plan"]["searches"] == [
        {"query": "general AI update", "intent": "general"},
        {"query": "official AI release", "intent": "official"},
        {"query": "AI research paper", "intent": "research"},
    ]

    replayable_spans = [
        SpanRecord.model_validate_json(span.model_dump_json()) for span in spans
    ]
    assert replayable_spans == spans
    run_span = next(span for span in spans if span.name == "agent.run")
    assert run_span.output["result"] == output.result.model_dump(mode="json")
