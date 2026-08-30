"""Tests for the fallback atomic research policy."""

import asyncio
import json

import pytest

from banso.agent.action import AgentAction, AgentActionType, RetrievalRoute
from banso.agent.policies.llm_atomic_policy import LLMAtomicPolicy
from banso.agent.policies.llm_news_policy import LLMPolicyError
from banso.agent.research_context import ResearchContextBuilder
from banso.agent.state import (
    AgentState,
    ExecutionBudget,
    SearchResultState,
    UserQuery,
)
from banso.artifacts.store import InMemoryArtifactStore
from banso.llm.models import LLMRequest, LLMResponse
from banso.retrieval.models import SearchResult


class StaticClient:
    def __init__(self, output: dict[str, object]) -> None:
        self.output = output
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(content=json.dumps(self.output))


def _policy(
    output: dict[str, object],
    *,
    store: InMemoryArtifactStore | None = None,
    routes: list[RetrievalRoute] | None = None,
) -> tuple[LLMAtomicPolicy, StaticClient]:
    client = StaticClient(output)
    policy = LLMAtomicPolicy(
        client,
        ResearchContextBuilder(
            store or InMemoryArtifactStore(),
            routes or [RetrievalRoute.WEB],
        ),
    )
    return policy, client


def test_selects_research_without_exposing_candidates() -> None:
    store = InMemoryArtifactStore()
    result = SearchResult(
        id="result",
        title="Candidate",
        url="https://example.com/candidate",
    )
    store.put(result)
    state = AgentState(
        query=UserQuery(text="question"),
        search_results={
            result.id: SearchResultState(retrieval_route=RetrievalRoute.WEB)
        },
    )
    policy, client = _policy(
        {
            "type": "research",
            "params": {"query": " focused query ", "route": "web"},
            "rationale": "Need evidence.",
        },
        store=store,
    )

    action = asyncio.run(policy.select_action(state))

    assert action == AgentAction(
        type=AgentActionType.RESEARCH,
        params={"query": "focused query", "route": "web"},
        rationale="Need evidence.",
    )
    request = client.requests[0]
    context = json.loads(request.messages[1].content)["context"]
    assert "candidate_results" not in context["retrieval_context"]
    assert request.metadata == {"trace": {"operation": "atomic_policy.select_action"}}
    assert '"type": "<research|rewrite_notes|stop>"' in request.messages[0].content


def test_rejects_disabled_research_route() -> None:
    policy, _ = _policy(
        {
            "type": "research",
            "params": {"query": "query", "route": "local"},
            "rationale": "Try local.",
        }
    )

    with pytest.raises(LLMPolicyError, match="disabled route"):
        asyncio.run(
            policy.select_action(AgentState(query=UserQuery(text="question")))
        )


def test_research_budget_removes_research_action() -> None:
    policy, _ = _policy(
        {
            "type": "research",
            "params": {"query": "query", "route": "web"},
            "rationale": "Need evidence.",
        }
    )
    state = AgentState(
        query=UserQuery(text="question"),
        budget=ExecutionBudget(max_researches=0),
    )

    with pytest.raises(LLMPolicyError) as caught:
        asyncio.run(policy.select_action(state))

    assert caught.value.reason == "invalid_action"
