"""Tests for search planning as an independently executable action."""

import asyncio

from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    AgentAction,
    AgentActionType,
    AgentRuntime,
    AgentState,
    DefaultStateReducer,
    ExecutionBudget,
    Observation,
    PlannedSearch,
    SearchPlan,
    UserQuery,
)
from banso.documents import FakeDocumentReader, FakeEvidenceExtractor
from banso.executors import NewsActionExecutor
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import (
    FakeRetrievalProvider,
    OriginalQueryPlanner,
    SearchPlanningRequest,
)
from banso.synthesis import FakeSynthesizer
from banso.tracing import AgentTrace


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
        document_reader=FakeDocumentReader(),
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
        planner.plan(SearchPlanningRequest(query=query, max_searches=1))
    )
    empty_plan = asyncio.run(
        planner.plan(SearchPlanningRequest(query=query, max_searches=0))
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
        budget=ExecutionBudget(max_searches=2),
    )

    observation = asyncio.run(
        executor.execute(
            AgentAction(type=AgentActionType.PLAN_SEARCH),
            state,
        )
    )

    assert planner.requests == [
        SearchPlanningRequest(query=state.query, max_searches=2)
    ]
    assert observation == Observation(
        action_type=AgentActionType.PLAN_SEARCH,
        data={"search_plan": plan.model_dump(mode="json")},
    )


def test_reducer_writes_search_plan_without_mutating_input_state() -> None:
    state = AgentState(query=UserQuery(text="latest AI news"))
    action = AgentAction(type=AgentActionType.PLAN_SEARCH)
    observation = Observation(
        action_type=AgentActionType.PLAN_SEARCH,
        data={
            "search_plan": {
                "searches": [{"query": "AI release", "intent": "official"}]
            }
        },
    )

    next_state = DefaultStateReducer().apply(state, action, observation)

    assert state.search_plan is None
    assert state.current_step == 0
    assert next_state.search_plan == SearchPlan(
        searches=[PlannedSearch(query="AI release", intent="official")]
    )
    assert next_state.last_action == AgentActionType.PLAN_SEARCH
    assert next_state.current_step == 1


def test_reducer_writes_final_answer_without_mutating_input_state() -> None:
    state = AgentState(query=UserQuery(text="latest AI news"))
    action = AgentAction(type=AgentActionType.SYNTHESIZE)
    observation = Observation(
        action_type=AgentActionType.SYNTHESIZE,
        data={"final_answer": "Synthesized answer."},
    )

    next_state = DefaultStateReducer().apply(state, action, observation)

    assert state.final_answer is None
    assert next_state.final_answer == "Synthesized answer."


async def _run_runtime_with_bounded_plan():
    plan = SearchPlan(
        searches=[
            PlannedSearch(query="general AI update"),
            PlannedSearch(query="official AI release", intent="official"),
            PlannedSearch(query="AI research paper", intent="research"),
        ]
    )
    planner = RecordingSearchQueryPlanner(plan)
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=_news_executor(planner),
    )

    return await runtime.run(
        AgentState(
            query=UserQuery(text="latest AI news"),
            budget=ExecutionBudget(max_searches=2),
        )
    )


def test_runtime_executes_bounded_search_plan_in_order() -> None:
    output = asyncio.run(_run_runtime_with_bounded_plan())

    assert [step.action.type for step in output.trace.steps] == [
        AgentActionType.PLAN_SEARCH,
        AgentActionType.SEARCH,
        AgentActionType.SEARCH,
        AgentActionType.READ_DOCUMENT,
        AgentActionType.EXTRACT_EVIDENCE,
        AgentActionType.SYNTHESIZE,
        AgentActionType.STOP,
    ]
    assert output.result.state.search_queries == [
        "general AI update",
        "official AI release",
    ]
    assert output.trace.steps[0].observation.data["search_plan"]["searches"] == [
        {"query": "general AI update", "intent": "general"},
        {"query": "official AI release", "intent": "official"},
        {"query": "AI research paper", "intent": "research"},
    ]

    replayable_trace = AgentTrace.model_validate_json(output.trace.model_dump_json())
    assert replayable_trace.steps[0].action.type == AgentActionType.PLAN_SEARCH
    assert replayable_trace.final_result is not None
    assert replayable_trace.final_result.state.search_plan == (
        output.result.state.search_plan
    )
