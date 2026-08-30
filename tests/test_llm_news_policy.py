"""Tests for the default LLM-backed news policy."""

import asyncio
import json

import pytest

from banso.agent.action import AgentAction, AgentActionType, RetrievalRoute
from banso.agent.policies.llm_news_policy import LLMNewsPolicy, LLMPolicyError
from banso.agent.research_context import ResearchContextBuilder
from banso.agent.state import (
    AgentState,
    ExecutionBudget,
    SearchResultState,
    UserQuery,
)
from banso.artifacts.store import InMemoryArtifactStore
from banso.llm.errors import LLMError
from banso.llm.models import LLMRequest, LLMResponse
from banso.retrieval.models import SearchResult


Output = dict[str, object] | str | Exception


class StubClient:
    def __init__(self, *outputs: Output) -> None:
        self.outputs = outputs
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        output = self.outputs[min(len(self.requests), len(self.outputs) - 1)]
        self.requests.append(request)
        if isinstance(output, Exception):
            raise output
        content = output if isinstance(output, str) else json.dumps(output)
        return LLMResponse(content=content)


def _policy(
    *outputs: Output,
    store: InMemoryArtifactStore | None = None,
) -> tuple[LLMNewsPolicy, StubClient]:
    client = StubClient(*outputs)
    policy = LLMNewsPolicy(
        client,
        ResearchContextBuilder(
            store or InMemoryArtifactStore(),
            [RetrievalRoute.WEB],
        ),
    )
    return policy, client


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


def test_selects_search_and_exposes_candidates() -> None:
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
    context = json.loads(request.messages[1].content)["context"]
    assert context["retrieval_context"]["candidate_results"] == []
    assert request.response_format == {"type": "json_object"}
    assert request.metadata == {"trace": {"operation": "news_policy.select_action"}}
    assert '"type": "<search|rewrite_notes|stop>"' in request.messages[0].content


def test_selects_available_candidates_for_read() -> None:
    state, store = _state_with_candidates(2)
    policy, client = _policy(
        {
            "type": "read",
            "params": {"search_result_refs": ["C2", "C1"]},
            "rationale": "Read relevant candidates.",
        },
        store=store,
    )

    action = asyncio.run(policy.select_action(state))

    assert action.params == {"search_result_refs": ["C2", "C1"]}
    context = json.loads(client.requests[0].messages[1].content)["context"]
    candidates = context["retrieval_context"]["candidate_results"]
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
        store=store,
    )

    with pytest.raises(LLMPolicyError, match=message):
        asyncio.run(policy.select_action(state))


def test_retries_invalid_output_once_with_the_same_request() -> None:
    policy, client = _policy(
        "not json",
        {
            "type": "search",
            "params": {"query": "query", "route": "web"},
            "rationale": "Need candidates.",
        },
    )

    action = asyncio.run(
        policy.select_action(AgentState(query=UserQuery(text="question")))
    )

    assert action.type == AgentActionType.SEARCH
    assert len(client.requests) == 2
    assert client.requests[0] is client.requests[1]


def test_second_invalid_output_is_preserved() -> None:
    policy, client = _policy("first invalid", "second invalid")

    with pytest.raises(LLMPolicyError) as caught:
        asyncio.run(
            policy.select_action(AgentState(query=UserQuery(text="question")))
        )

    assert caught.value.reason == "invalid_json"
    assert caught.value.raw_output == "second invalid"
    assert len(client.requests) == 2


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        (
            {
                "type": "search",
                "params": {"query": "query", "route": "web"},
                "rationale": "Search.",
                "extra": True,
            },
            "invalid_schema",
        ),
        (
            {
                "type": "search",
                "params": {"query": "query", "route": "web"},
                "rationale": " ",
            },
            "invalid_params",
        ),
        (
            {
                "type": "research",
                "params": {"query": "query", "route": "web"},
                "rationale": "Research.",
            },
            "invalid_action",
        ),
    ],
)
def test_rejects_invalid_action_outputs(output: dict, reason: str) -> None:
    policy, _ = _policy(output)

    with pytest.raises(LLMPolicyError) as caught:
        asyncio.run(
            policy.select_action(AgentState(query=UserQuery(text="question")))
        )

    assert caught.value.reason == reason


def test_provider_error_is_not_retried() -> None:
    policy, client = _policy(LLMError(RuntimeError("provider failed")))

    with pytest.raises(LLMPolicyError) as caught:
        asyncio.run(
            policy.select_action(AgentState(query=UserQuery(text="question")))
        )

    assert caught.value.reason == "llm_error"
    assert len(client.requests) == 1


def test_last_step_without_evidence_exposes_only_stop() -> None:
    policy, client = _policy(
        {"type": "stop", "params": {}, "rationale": "No evidence."}
    )
    state = AgentState(
        query=UserQuery(text="question"),
        current_step=1,
        budget=ExecutionBudget(max_steps=2),
    )

    action = asyncio.run(policy.select_action(state))

    assert action.type == AgentActionType.STOP
    assert '"type": "<stop>"' in client.requests[0].messages[0].content
