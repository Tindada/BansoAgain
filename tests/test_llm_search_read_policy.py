"""Tests for the LLM-backed search/read policy."""

import asyncio
import json

import pytest

from banso.artifacts.store import InMemoryArtifactStore
from banso.agent.action import AgentAction, AgentActionType, RetrievalRoute
from banso.agent.policies.llm_news_policy import LLMPolicyError
from banso.agent.policies.llm_search_read_policy import LLMSearchReadPolicy
from banso.agent.research_context import ResearchContextBuilder
from banso.agent.state import (
    AgentState,
    ExecutionBudget,
    SearchResultState,
    UserQuery,
)
from banso.llm.models import LLMRequest, LLMResponse
from banso.retrieval.models import SearchResult


class StaticClient:
    def __init__(self, output: dict) -> None:
        self.output = output
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(content=json.dumps(self.output))


def _policy(
    output: dict,
    store: InMemoryArtifactStore | None = None,
) -> tuple[LLMSearchReadPolicy, StaticClient]:
    client = StaticClient(output)
    return (
        LLMSearchReadPolicy(
            client,
            ResearchContextBuilder(
                store or InMemoryArtifactStore(),
                [RetrievalRoute.WEB],
            ),
        ),
        client,
    )


def _state_with_candidates(
    count: int,
    *,
    max_results_per_read: int = 10,
) -> tuple[AgentState, InMemoryArtifactStore]:
    store = InMemoryArtifactStore()
    state = AgentState(
        query=UserQuery(text="question"),
        budget=ExecutionBudget(max_results_per_research=max_results_per_read),
    )
    for index in range(1, count + 1):
        result = SearchResult(
            id=f"result-{index}",
            title=f"Result {index}",
            url=f"https://example.com/{index}",
        )
        store.put(result)
        state.search_results[result.id] = SearchResultState(
            retrieval_route=RetrievalRoute.WEB
        )
    return state, store


def test_selects_search_and_exposes_search_read_prompt() -> None:
    policy, client = _policy(
        {
            "type": "search",
            "params": {"query": "  focused query  ", "route": "web"},
            "rationale": "Find candidates.",
        }
    )

    action = asyncio.run(
        policy.select_action(AgentState(query=UserQuery(text="question")))
    )

    assert action == AgentAction(
        type=AgentActionType.SEARCH,
        params={"query": "focused query", "route": "web"},
        rationale="Find candidates.",
    )
    request = client.requests[0]
    prompt = json.loads(request.messages[1].content)
    assert prompt["context"]["retrieval_context"]["candidate_results"] == []
    assert request.metadata == {
        "trace": {"operation": "search_read_policy.select_action"}
    }
    assert '"type": "<search|rewrite_notes|stop>"' in request.messages[0].content


def test_selects_available_candidates_for_read() -> None:
    state, store = _state_with_candidates(2)
    policy, client = _policy(
        {
            "type": "read",
            "params": {"search_result_refs": ["C2", "C1"]},
            "rationale": "Read relevant candidates.",
        },
        store,
    )

    action = asyncio.run(policy.select_action(state))

    assert action.params == {"search_result_refs": ["C2", "C1"]}
    candidates = json.loads(client.requests[0].messages[1].content)["context"][
        "retrieval_context"
    ]["candidate_results"]
    assert [candidate["candidate_ref"] for candidate in candidates] == ["C1", "C2"]


@pytest.mark.parametrize(
    ("refs", "limit", "message"),
    [
        (["C2"], 10, "unavailable candidate ref"),
        (["C1", "C2"], 1, "per-read result limit"),
    ],
)
def test_rejects_invalid_read_refs(
    refs: list[str],
    limit: int,
    message: str,
) -> None:
    state, store = _state_with_candidates(
        1 if refs == ["C2"] else 2,
        max_results_per_read=limit,
    )
    policy, _ = _policy(
        {
            "type": "read",
            "params": {"search_result_refs": refs},
            "rationale": "Read.",
        },
        store,
    )

    with pytest.raises(LLMPolicyError, match=message):
        asyncio.run(policy.select_action(state))
