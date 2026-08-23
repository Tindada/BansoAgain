"""Tests for the LLM-backed atomic research policy."""

import asyncio
import json

import pytest

from banso.artifacts.store import InMemoryArtifactStore
from banso.agent.action import (
    AgentAction,
    AgentActionType,
    RetrievalRoute,
)
from banso.agent.observation import (
    CompletedResearchObservation,
    ResearchObservation,
    RetrievalFailedResearchObservation,
)
from banso.agent.reducer import DefaultStateReducer
from banso.agent.state import AgentState, DocumentState, ExecutionBudget, UserQuery
from banso.documents.models import Document
from banso.llm.errors import LLMError
from banso.llm.models import LLMRequest, LLMResponse
from banso.agent.policies.llm_news_policy import LLMNewsPolicy, LLMPolicyError
from banso.retrieval.models import (
    RetrievalFilterReport,
    SearchResultMergeReport,
    SearchResultSelectionReport,
    SourceClassificationReport,
)
from banso.agent.research_context import ResearchContextBuilder


class StaticClient:
    def __init__(self, output: dict | str) -> None:
        self.output = output
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        content = self.output if isinstance(self.output, str) else json.dumps(self.output)
        return LLMResponse(content=content)


class RaisingClient:
    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise LLMError(RuntimeError("provider failed"))


def _policy(
    output: dict | str,
    store: InMemoryArtifactStore | None = None,
    routes: list[RetrievalRoute] | None = None,
) -> tuple[LLMNewsPolicy, StaticClient]:
    client = StaticClient(output)
    policy = LLMNewsPolicy(
        client,
        ResearchContextBuilder(
            store or InMemoryArtifactStore(),
            routes or [RetrievalRoute.WEB],
        ),
    )
    return policy, client


def _empty_research(
    *,
    query: str = "query",
    route: RetrievalRoute = RetrievalRoute.WEB,
) -> CompletedResearchObservation:
    result_ids: list[str] = []
    return CompletedResearchObservation(
        query=query,
        route=route,
        search_result_ids=result_ids,
        search_result_index_updates={},
        search_result_merge_report=SearchResultMergeReport(
            candidate_count=len(result_ids),
            new_result_count=len(result_ids),
            reused_result_count=0,
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
        selection_report=SearchResultSelectionReport(
            candidate_ids=result_ids,
            selected_ids=result_ids,
        ),
        fetch_outcomes=[],
        document_index_updates={},
        extraction_outcomes=[],
    )


def _apply_research(
    state: AgentState,
    observation: ResearchObservation,
) -> AgentState:
    return DefaultStateReducer().apply(
        state,
        AgentAction(
            type=AgentActionType.RESEARCH,
            params={"query": observation.query, "route": observation.route.value},
        ),
        observation,
    )


def _curation_state_and_store() -> tuple[AgentState, InMemoryArtifactStore]:
    store = InMemoryArtifactStore()
    for document_id in ("active", "shelved"):
        store.put(
            Document(
                id=document_id,
                url=f"https://example.com/{document_id}",
                title=document_id.title(),
                text=document_id,
            )
        )
    return (
        AgentState(
            query=UserQuery(text="question"),
            documents={
                "active": DocumentState(lifecycle_status="active"),
                "shelved": DocumentState(lifecycle_status="shelved"),
            },
        ),
        store,
    )


def test_selects_research_with_an_enabled_route() -> None:
    policy, client = _policy(
        {
            "type": "research",
            "params": {
                "query": "  focused query  ",
                "route": "web",
                "source_domains": [],
            },
            "rationale": "Need more evidence.",
        }
    )

    action = asyncio.run(
        policy.select_action(AgentState(query=UserQuery(text="question")))
    )

    assert action.type == AgentActionType.RESEARCH
    assert action.params == {"query": "focused query", "route": "web"}
    prompt = json.loads(client.requests[0].messages[1].content)
    assert set(prompt) == {"context"}
    assert prompt["context"]["enabled_routes"] == ["web"]
    assert "candidate_results" not in prompt["context"]
    assert "candidate_documents" not in prompt["context"]
    system_prompt = client.requests[0].messages[0].content
    assert "research_refs" in system_prompt
    assert "\n  research:\n    Instruction:" in system_prompt
    assert "only valid for web" in system_prompt
    assert (
        '    Params: {"query": "<non-empty string>", "route": "web|local", '
        '"source_domains": ["<bare domain>"]}'
    ) in system_prompt
    output_format = system_prompt.split("Output format:\n", 1)[1].splitlines()[0]
    assert output_format == (
        '{"type": "<research|stop>", "params": <matching Params object>, '
        '"rationale": "<brief decision reason>"}'
    )


def test_selects_web_research_with_source_domains() -> None:
    policy, _ = _policy(
        {
            "type": "research",
            "params": {
                "query": "focused query",
                "route": "web",
                "source_domains": [" X.COM ", "twitter.com"],
            },
            "rationale": "Search the requested platform.",
        }
    )

    action = asyncio.run(
        policy.select_action(AgentState(query=UserQuery(text="question")))
    )

    assert action.params == {
        "query": "focused query",
        "route": "web",
        "source_domains": ["x.com", "twitter.com"],
    }


def test_same_query_is_allowed_on_a_different_route() -> None:
    state = _apply_research(
        AgentState(query=UserQuery(text="question")),
        _empty_research(route=RetrievalRoute.WEB),
    )
    policy, _ = _policy(
        {
            "type": "research",
            "params": {"query": "query", "route": "local"},
            "rationale": "Use local evidence.",
        },
        routes=[RetrievalRoute.WEB, RetrievalRoute.LOCAL],
    )

    action = asyncio.run(policy.select_action(state))

    assert action.params["route"] == "local"


def test_retrieval_failure_is_visible_to_policy_without_external_message() -> None:
    state = _apply_research(
        AgentState(query=UserQuery(text="question")),
        RetrievalFailedResearchObservation(
            query="query",
            route=RetrievalRoute.WEB,
            source_domains=["x.com"],
            provider="tavily",
            reason="http_status",
            status_code=400,
            message="untrusted provider response",
            source_error_type="HTTPStatusError",
            retryable=False,
            attempt_count=1,
        ),
    )
    policy, client = _policy(
        {"type": "stop", "params": {}, "rationale": "Cannot progress."}
    )

    asyncio.run(policy.select_action(state))

    prompt = json.loads(client.requests[0].messages[1].content)
    history = prompt["context"]["research_history"][0]
    assert history["research_ref"] == "R1"
    assert history["status"] == "retrieval_failed"
    assert history["source_domains"] == ["x.com"]
    assert history["reason"] == "http_status"
    assert history["status_code"] == 400
    assert "untrusted provider response" not in client.requests[0].messages[1].content


def test_rejects_disabled_route_and_invalid_output() -> None:
    disabled_policy, _ = _policy(
        {
            "type": "research",
            "params": {"query": "query", "route": "local"},
            "rationale": "Try local.",
        }
    )
    with pytest.raises(LLMPolicyError, match="disabled route"):
        asyncio.run(
            disabled_policy.select_action(AgentState(query=UserQuery(text="question")))
        )

    invalid_policy, _ = _policy("not json")
    with pytest.raises(LLMPolicyError) as caught:
        asyncio.run(
            invalid_policy.select_action(AgentState(query=UserQuery(text="question")))
        )
    assert caught.value.reason == "invalid_json"


@pytest.mark.parametrize(
    ("output", "reason"),
    [
        (
            {
                "type": "research",
                "params": {"query": "query", "route": "web"},
                "rationale": "Research.",
                "extra": True,
            },
            "invalid_schema",
        ),
        (
            {
                "type": "research",
                "params": {"query": "query", "route": "web"},
                "rationale": " ",
            },
            "invalid_params",
        ),
        (
            {
                "type": "stop",
                "params": {"unsupported": True},
                "rationale": "Stop.",
            },
            "invalid_params",
        ),
    ],
)
def test_rejects_invalid_action_outputs(output: dict, reason: str) -> None:
    policy, _ = _policy(output)

    with pytest.raises(LLMPolicyError) as caught:
        asyncio.run(policy.select_action(AgentState(query=UserQuery(text="question"))))

    assert caught.value.reason == reason


@pytest.mark.parametrize(
    ("active_refs", "message"),
    [
        (["D3"], "unknown"),
        (["D1", "D1"], "unique"),
        (["D1"], "must change"),
    ],
)
def test_rejects_invalid_curation_refs(
    active_refs: list[str],
    message: str,
) -> None:
    state, store = _curation_state_and_store()
    policy, _ = _policy(
        {
            "type": "curate_evidence",
            "params": {"active_document_refs": active_refs},
            "rationale": "Curate.",
        },
        store=store,
    )

    with pytest.raises(LLMPolicyError, match=message):
        asyncio.run(policy.select_action(state))


def test_wraps_llm_errors() -> None:
    policy = LLMNewsPolicy(
        RaisingClient(),
        ResearchContextBuilder(
            InMemoryArtifactStore(),
            [RetrievalRoute.WEB],
        ),
    )

    with pytest.raises(LLMPolicyError) as caught:
        asyncio.run(policy.select_action(AgentState(query=UserQuery(text="question"))))

    assert caught.value.reason == "llm_error"


def test_research_budget_removes_research_from_available_actions() -> None:
    state = AgentState(
        query=UserQuery(text="question"),
        budget=ExecutionBudget(max_researches=1),
    )
    state = _apply_research(state, _empty_research())
    policy, _ = _policy(
        {
            "type": "research",
            "params": {"query": "new", "route": "web"},
            "rationale": "More.",
        }
    )

    with pytest.raises(LLMPolicyError) as caught:
        asyncio.run(policy.select_action(state))
    assert caught.value.reason == "invalid_action"


def test_curation_maps_document_refs_to_lifecycle_transitions() -> None:
    store = InMemoryArtifactStore()
    active = Document(id="active", url="https://example.com/a", title="A", text="A")
    shelved = Document(id="shelved", url="https://example.com/s", title="S", text="S")
    store.put(active)
    store.put(shelved)
    state = AgentState(
        query=UserQuery(text="question"),
        documents={
            "active": DocumentState(lifecycle_status="active"),
            "shelved": DocumentState(lifecycle_status="shelved"),
        },
    )
    policy, _ = _policy(
        {
            "type": "curate_evidence",
            "params": {"active_document_refs": ["D2"]},
            "rationale": "Prefer the second source.",
        },
        store=store,
    )

    action = asyncio.run(policy.select_action(state))

    assert action.params == {
        "shelve_document_ids": ["active"],
        "reactivate_document_ids": ["shelved"],
    }


def test_last_step_exposes_only_finish_or_stop() -> None:
    store = InMemoryArtifactStore()
    document = Document(id="document", url="https://example.com", title="D", text="D")
    store.put(document)
    state = AgentState(
        query=UserQuery(text="question"),
        current_step=1,
        budget=ExecutionBudget(max_steps=2),
        documents={
            "document": DocumentState(lifecycle_status="active")
        },
    )
    policy, client = _policy(
        {"type": "finish", "params": {}, "rationale": "Enough evidence."},
        store=store,
    )

    action = asyncio.run(policy.select_action(state))

    assert action.type == AgentActionType.FINISH
    system_prompt = client.requests[0].messages[0].content
    assert '"type": "<finish|stop>"' in system_prompt


def test_last_step_without_active_evidence_exposes_only_stop() -> None:
    state = AgentState(
        query=UserQuery(text="question"),
        current_step=1,
        budget=ExecutionBudget(max_steps=2),
    )
    policy, client = _policy(
        {"type": "stop", "params": {}, "rationale": "No evidence."}
    )

    action = asyncio.run(policy.select_action(state))

    assert action.type == AgentActionType.STOP
    system_prompt = client.requests[0].messages[0].content
    assert '"type": "<stop>"' in system_prompt
