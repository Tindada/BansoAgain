"""Tests for the LLM-backed search query planner."""

import asyncio
from datetime import datetime, timezone

import pytest

from banso.apps import real_news
from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    AgentActionType,
    AgentRuntime,
    AgentState,
    ExecutionBudget,
    SearchPlan,
    UserQuery,
)
from banso.documents import FakeDocumentReader, FakeEvidenceExtractor
from banso.executors import NewsActionExecutor
from banso.llm import FakeLLMClient, LLMMessageRole
from banso.llm.tracing import TracingLLMClient
from banso.policies import LLMNewsPolicy, NewsRuleBasedPolicy
from banso.retrieval import (
    FakeRetrievalProvider,
    LLMSearchQueryPlanner,
    SearchPlanningError,
    SearchPlanningRequest,
)
from banso.synthesis import FakeSynthesizer

REFERENCE_TIME = datetime(2026, 7, 24, 8, 30, tzinfo=timezone.utc)


def test_llm_search_query_planner_builds_bounded_distinct_plan() -> None:
    client = FakeLLMClient(
        content=(
            '{"searches": ['
            '{"query": " AI product releases ", "intent": " official news "},'
            '{"query": "ai PRODUCT releases", "intent": "duplicate"},'
            '{"query": "AI research results", "intent": "research"},'
            '{"query": "AI regulation updates", "intent": "policy"}'
            "]}"
        )
    )
    planner = LLMSearchQueryPlanner(client=client, model="planner-model")
    request = SearchPlanningRequest(
        query=UserQuery(
            text="What happened in AI this week?",
            language="en",
            region="US",
            time_range="past week",
        ),
        reference_time=REFERENCE_TIME,
        max_searches=2,
    )

    plan = asyncio.run(planner.plan(request))

    assert plan == SearchPlan.model_validate(
        {
            "searches": [
                {"query": "AI product releases", "intent": "official news"},
                {"query": "AI research results", "intent": "research"},
            ]
        }
    )
    assert len(client.requests) == 1
    llm_request = client.requests[0]
    assert llm_request.model == "planner-model"
    assert llm_request.temperature == 0.0
    assert [message.role for message in llm_request.messages] == [
        LLMMessageRole.SYSTEM,
        LLMMessageRole.USER,
    ]
    user_prompt = llm_request.messages[1].content
    assert "What happened in AI this week?" in user_prompt
    assert "Reference time: 2026-07-24T08:30:00+00:00" in user_prompt
    assert "Language: en" in user_prompt
    assert "Region: US" in user_prompt
    assert "Time range: past week" in user_prompt
    assert "Maximum searches: 2" in user_prompt
    assert '"searches"' in user_prompt


def test_llm_search_query_planner_skips_llm_for_zero_budget() -> None:
    client = FakeLLMClient(content='{"searches": []}')
    planner = LLMSearchQueryPlanner(client=client)

    plan = asyncio.run(
        planner.plan(
            SearchPlanningRequest(
                query=UserQuery(text="latest AI news"),
                reference_time=REFERENCE_TIME,
                max_searches=0,
            )
        )
    )

    assert plan == SearchPlan()
    assert client.requests == []


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("not json", "invalid_json"),
        ('[{"query": "AI news"}]', "invalid_schema"),
        ('{"searches": [{"query": " ", "intent": "general"}]}', "invalid_schema"),
        ('{"searches": []}', "empty_plan"),
    ],
)
def test_llm_search_query_planner_rejects_invalid_responses(
    content: str,
    reason: str,
) -> None:
    planner = LLMSearchQueryPlanner(client=FakeLLMClient(content=content))

    with pytest.raises(SearchPlanningError) as caught:
        asyncio.run(
            planner.plan(
                SearchPlanningRequest(
                    query=UserQuery(text="latest AI news"),
                    reference_time=REFERENCE_TIME,
                    max_searches=2,
                )
            )
        )

    assert caught.value.reason == reason


async def _run_news_runtime_with_llm_plan():
    client = FakeLLMClient(
        content=(
            '{"searches": ['
            '{"query": "official AI releases", "intent": "official"},'
            '{"query": "recent AI research", "intent": "research"}'
            "]}"
        )
    )
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=NewsActionExecutor(
            store=InMemoryArtifactStore(),
            retrieval_provider=FakeRetrievalProvider(),
            document_reader=FakeDocumentReader(),
            evidence_extractor=FakeEvidenceExtractor(),
            synthesizer=FakeSynthesizer(),
            search_query_planner=LLMSearchQueryPlanner(client=client),
        ),
    )
    return await runtime.run(
        AgentState(
            query=UserQuery(text="latest AI news"),
            budget=ExecutionBudget(max_searches=2),
        )
    )


def test_news_runtime_executes_llm_generated_search_plan() -> None:
    output = asyncio.run(_run_news_runtime_with_llm_plan())

    assert [
        entry.observation.data["search_queries"][0]
        for entry in output.result.state.action_history
        if entry.action_type == AgentActionType.SEARCH
    ] == [
        "official AI releases",
        "recent AI research",
    ]
    assert output.result.state.action_history[0].observation.data["search_plan"] == {
        "searches": [
            {"query": "official AI releases", "intent": "official"},
            {"query": "recent AI research", "intent": "research"},
        ]
    }


def test_real_news_runtime_defaults_to_rule_policy_and_reuses_llm_clients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_client = FakeLLMClient()
    external_client = FakeLLMClient()
    monkeypatch.delenv("BANSO_NEWS_POLICY", raising=False)
    monkeypatch.setattr(
        real_news,
        "build_vllm_llm_client_from_env",
        lambda: local_client,
    )
    monkeypatch.setattr(
        real_news,
        "build_external_llm_client_from_env",
        lambda: external_client,
    )
    monkeypatch.setattr(
        real_news,
        "build_tavily_provider_from_env",
        FakeRetrievalProvider,
    )

    bundle = real_news.build_real_news_runtime()
    executor = bundle.runtime.executor

    assert isinstance(bundle.runtime.policy, NewsRuleBasedPolicy)
    assert isinstance(executor, NewsActionExecutor)
    assert isinstance(executor.search_query_planner, LLMSearchQueryPlanner)
    assert executor.search_query_planner.client is executor.synthesizer.client
    assert isinstance(executor.search_query_planner.client, TracingLLMClient)
    assert executor.search_query_planner.client.client is external_client
    assert executor.evidence_extractor.client.client.client is local_client
    assert isinstance(executor.synthesizer.client, TracingLLMClient)
    assert executor.synthesizer.client.client is external_client


def test_real_news_runtime_builds_llm_policy_with_shared_store_and_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_client = FakeLLMClient()
    external_client = FakeLLMClient()
    monkeypatch.setenv("BANSO_NEWS_POLICY", "llm")
    monkeypatch.setattr(
        real_news,
        "build_vllm_llm_client_from_env",
        lambda: local_client,
    )
    monkeypatch.setattr(
        real_news,
        "build_external_llm_client_from_env",
        lambda: external_client,
    )
    monkeypatch.setattr(
        real_news,
        "build_tavily_provider_from_env",
        FakeRetrievalProvider,
    )

    bundle = real_news.build_real_news_runtime()
    policy = bundle.runtime.policy
    executor = bundle.runtime.executor

    assert isinstance(policy, LLMNewsPolicy)
    assert isinstance(executor, NewsActionExecutor)
    assert policy.client is executor.evidence_extractor.client
    assert isinstance(policy.client, TracingLLMClient)
    assert policy.client.client.client is local_client
    assert executor.search_query_planner.client is executor.synthesizer.client
    assert executor.search_query_planner.client.client is external_client
    assert policy.context_builder.store is bundle.store
    assert executor.store is bundle.store
    assert isinstance(executor.synthesizer.client, TracingLLMClient)
    assert executor.synthesizer.client.client is external_client


def test_real_news_runtime_rejects_unknown_policy_before_building_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BANSO_NEWS_POLICY", "unknown")

    with pytest.raises(
        RuntimeError,
        match="BANSO_NEWS_POLICY must be 'rule_based' or 'llm', got 'unknown'",
    ):
        real_news.build_real_news_runtime()
