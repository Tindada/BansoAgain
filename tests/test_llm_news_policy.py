"""Tests for the LLM-backed news policy."""

import asyncio
import json

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    ActionHistoryEntry,
    AgentAction,
    AgentActionType,
    AgentState,
    ExecutionBudget,
    Observation,
    SearchPlan,
    UserQuery,
)
from banso.documents import Document, EvidenceItem
from banso.llm import (
    FakeLLMClient,
    LLMError,
    LLMMessageRole,
    LLMRequest,
    LLMResponse,
)
from banso.policies import LLMNewsPolicy, LLMPolicyError, NewsPolicyStateViewBuilder
from banso.retrieval import SearchResult


def _populated_state() -> tuple[InMemoryArtifactStore, AgentState]:
    store = InMemoryArtifactStore()
    store.put(
        SearchResult(
            id="result-1",
            title="Result",
            url="https://example.com/result",
            snippet="visible snippet",
            metadata={"hidden_search_metadata": True},
        )
    )
    store.put(
        Document(
            id="document-1",
            title="Document",
            url="https://example.com/document",
            text="visible document preview hidden_document_tail",
            metadata={"hidden_document_metadata": True},
        )
    )
    store.put(
        EvidenceItem(
            id="evidence-1",
            document_id="document-1",
            claim="visible claim",
            supporting_text="hidden_supporting_text",
            source_url="https://example.com/document",
            metadata={"hidden_evidence_metadata": True},
        )
    )
    return store, AgentState(
        query=UserQuery(text="What happened?"),
        search_result_ids=["result-1"],
        document_ids=["document-1"],
        evidence_ids=["evidence-1"],
    )


def _select_action(
    content: str,
    state: AgentState,
    store: InMemoryArtifactStore,
) -> tuple[AgentAction, FakeLLMClient]:
    client = FakeLLMClient(content=content)
    policy = LLMNewsPolicy(client, NewsPolicyStateViewBuilder(store))
    return asyncio.run(policy.select_action(state)), client


@pytest.mark.parametrize(
    ("action_type", "params", "expected_params"),
    [
        (AgentActionType.PLAN_SEARCH, {}, {}),
        (
            AgentActionType.SEARCH,
            {"query": " AI news ", "intent": " latest updates "},
            {"query": "AI news", "intent": "latest updates"},
        ),
        (AgentActionType.READ_DOCUMENT, {}, {}),
        (AgentActionType.EXTRACT_EVIDENCE, {}, {}),
        (AgentActionType.SYNTHESIZE, {}, {}),
        (AgentActionType.STOP, {}, {}),
    ],
)
def test_selects_each_supported_action(
    action_type: AgentActionType,
    params: dict[str, str],
    expected_params: dict[str, str],
) -> None:
    store, state = _populated_state()
    content = json.dumps(
        {
            "type": action_type.value,
            "params": params,
            "rationale": " Choose the next useful step. ",
        }
    )

    action, client = _select_action(content, state, store)

    assert action == AgentAction(
        type=action_type,
        params=expected_params,
        rationale="Choose the next useful step.",
    )
    assert len(client.requests) == 1


def test_builds_request_from_bounded_policy_view_and_remaining_budget() -> None:
    store, state = _populated_state()
    state.current_step = 2
    state.budget = ExecutionBudget(max_steps=7, max_searches=3)
    state.action_history.append(
        ActionHistoryEntry(
            step_index=0,
            action_type=AgentActionType.SEARCH,
            params={"query": "first query"},
            observation=Observation(data={"search_result_ids": ["result-1"]}),
        )
    )
    state_before = state.model_copy(deep=True)
    client = FakeLLMClient(
        content='{"type":"stop","params":{},"rationale":"Enough information."}'
    )
    policy = LLMNewsPolicy(
        client,
        NewsPolicyStateViewBuilder(store, max_document_preview_chars=7),
        model="policy-model",
        temperature=0.2,
        max_tokens=128,
    )

    asyncio.run(policy.select_action(state))

    assert state == state_before
    request = client.requests[0]
    assert request.model == "policy-model"
    assert request.temperature == 0.2
    assert request.max_tokens == 128
    assert [message.role for message in request.messages] == [
        LLMMessageRole.SYSTEM,
        LLMMessageRole.USER,
    ]
    payload = json.loads(request.messages[1].content)
    assert payload["policy_state"]["state"]["query"]["text"] == "What happened?"
    assert payload["policy_state"]["documents"][0]["text_preview"] == "visible"
    assert payload["remaining_budget"] == {
        "remaining_step_count": 5,
        "executed_search_count": 1,
        "remaining_search_count": 2,
    }
    assert payload["available_actions"] == [
        "plan_search",
        "search",
        "read_document",
        "extract_evidence",
        "synthesize",
        "stop",
    ]
    prompt = request.messages[1].content
    assert "hidden_search_metadata" not in prompt
    assert "hidden_document_metadata" not in prompt
    assert "hidden_document_tail" not in prompt
    assert "hidden_supporting_text" not in prompt
    assert "hidden_evidence_metadata" not in prompt


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("not json", "invalid_json"),
        (
            '```json\n{"type":"stop","params":{},"rationale":"done"}\n```',
            "invalid_json",
        ),
        ('{"type":"unknown","params":{},"rationale":"done"}', "invalid_schema"),
        (
            '{"type":"stop","params":{},"rationale":"done","extra":true}',
            "invalid_schema",
        ),
        ('{"type":"stop","params":{},"rationale":"   "}', "invalid_params"),
    ],
)
def test_rejects_invalid_llm_output(content: str, reason: str) -> None:
    store, state = _populated_state()
    client = FakeLLMClient(content=content)
    policy = LLMNewsPolicy(client, NewsPolicyStateViewBuilder(store))
    state_before = state.model_copy(deep=True)

    with pytest.raises(LLMPolicyError) as exc_info:
        asyncio.run(policy.select_action(state))

    assert exc_info.value.reason == reason
    assert exc_info.value.raw_output == content
    assert "raw_output=" in str(exc_info.value)
    assert len(client.requests) == 1
    assert state == state_before


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"query": "   "},
        {"query": 123},
        {"query": "news", "intent": "   "},
        {"query": "news", "intent": 123},
        {"query": "news", "unexpected": "value"},
    ],
)
def test_rejects_invalid_search_params(params: dict[str, object]) -> None:
    store, state = _populated_state()
    content = json.dumps(
        {"type": "search", "params": params, "rationale": "Search for news."}
    )

    with pytest.raises(LLMPolicyError) as exc_info:
        _select_action(content, state, store)

    assert exc_info.value.reason == "invalid_params"


def test_rejects_params_for_non_search_action() -> None:
    store, state = _populated_state()

    with pytest.raises(LLMPolicyError) as exc_info:
        _select_action(
            '{"type":"stop","params":{"reason":"done"},'
            '"rationale":"Stop now."}',
            state,
            store,
        )

    assert exc_info.value.reason == "invalid_params"


def test_rejects_search_after_budget_is_exhausted() -> None:
    store, state = _populated_state()
    state.budget.max_searches = 1
    state.action_history.append(
        ActionHistoryEntry(
            step_index=0,
            action_type=AgentActionType.SEARCH,
            params={"query": "first query"},
            observation=Observation(),
        )
    )

    with pytest.raises(LLMPolicyError) as exc_info:
        _select_action(
            '{"type":"search","params":{"query":"second query"},'
            '"rationale":"Search again."}',
            state,
            store,
        )

    assert exc_info.value.reason == "invalid_action"


def test_rejects_repeated_search_query() -> None:
    store, state = _populated_state()
    state.action_history.append(
        ActionHistoryEntry(
            step_index=0,
            action_type=AgentActionType.SEARCH,
            params={"query": "AI News"},
            observation=Observation(),
        )
    )

    with pytest.raises(LLMPolicyError) as exc_info:
        _select_action(
            '{"type":"search","params":{"query":"  ai NEWS  "},'
            '"rationale":"Search again."}',
            state,
            store,
        )

    assert exc_info.value.reason == "invalid_action"


@pytest.mark.parametrize(
    ("action_type", "state_update"),
    [
        (AgentActionType.PLAN_SEARCH, "existing_plan"),
        (AgentActionType.READ_DOCUMENT, "no_results"),
        (AgentActionType.EXTRACT_EVIDENCE, "no_documents"),
        (AgentActionType.SYNTHESIZE, "no_sources"),
    ],
)
def test_rejects_action_without_required_state(
    action_type: AgentActionType,
    state_update: str,
) -> None:
    store, state = _populated_state()
    if state_update == "existing_plan":
        state.search_plan = SearchPlan()
    elif state_update == "no_results":
        state.search_result_ids = []
    elif state_update == "no_documents":
        state.document_ids = []
    else:
        state.document_ids = []
        state.evidence_ids = []
    content = json.dumps(
        {
            "type": action_type.value,
            "params": {},
            "rationale": "Take this action.",
        }
    )

    with pytest.raises(LLMPolicyError) as exc_info:
        _select_action(content, state, store)

    assert exc_info.value.reason == "invalid_action"


class _FailingLLMClient:
    def __init__(self) -> None:
        self.call_count = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        raise LLMError(RuntimeError("provider failed"))


def test_wraps_known_llm_error_without_retrying() -> None:
    store, state = _populated_state()
    client = _FailingLLMClient()
    policy = LLMNewsPolicy(client, NewsPolicyStateViewBuilder(store))

    with pytest.raises(LLMPolicyError) as exc_info:
        asyncio.run(policy.select_action(state))

    assert exc_info.value.reason == "llm_error"
    assert exc_info.value.raw_output is None
    assert isinstance(exc_info.value.__cause__, LLMError)
    assert client.call_count == 1
