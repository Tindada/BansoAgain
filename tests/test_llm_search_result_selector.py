"""Tests for LLM-backed search-result selection."""

import asyncio
import json

import pytest

from banso.agent.action import AgentAction, AgentActionType, RetrievalRoute
from banso.agent.observation import (
    CompletedResearchObservation,
    DocumentFetchFailure,
    FetchFailure,
)
from banso.agent.reducer import DefaultStateReducer
from banso.agent.research_context import ResearchContextBuilder
from banso.agent.selection.llm_selector import LLMSearchResultSelector
from banso.agent.selection.selector import (
    SearchResultSelectionError,
    SearchResultSelectionRequest,
)
from banso.agent.state import AgentState, UserQuery
from banso.artifacts.store import InMemoryArtifactStore
from banso.llm.fake import FakeLLMClient
from banso.retrieval.models import (
    RetrievalFilterReport,
    SearchResult,
    SearchResultMergeReport,
    SearchResultSelectionReport,
    SourceClassificationReport,
)


def _request(state: AgentState | None = None) -> SearchResultSelectionRequest:
    return SearchResultSelectionRequest(
        research_query="specific information gap",
        candidates=[
            SearchResult(
                id="result-1",
                title="Relevant result",
                url="https://example.com/relevant",
                snippet="Direct evidence",
                rank=1,
            ),
            SearchResult(
                id="result-2",
                title="Duplicate result",
                url="https://example.com/duplicate",
                rank=2,
            ),
        ],
        state=state or AgentState(query=UserQuery(text="overall question")),
    )


def _state_with_fetch_failure() -> AgentState:
    observation = CompletedResearchObservation(
        query="prior query",
        route=RetrievalRoute.WEB,
        search_result_ids=["failed-result"],
        retrieval_filter_report=RetrievalFilterReport(input_count=1, output_count=1),
        source_classification_report=SourceClassificationReport(
            input_count=1,
            recognized_count=0,
            unknown_count=1,
        ),
        search_result_merge_report=SearchResultMergeReport(
            candidate_count=1,
            new_result_count=1,
            reused_result_count=0,
        ),
        selection_report=SearchResultSelectionReport(
            candidate_ids=["failed-result"],
            selected_ids=["failed-result"],
        ),
        fetch_outcomes=[
            FetchFailure(
                search_result_id="failed-result",
                failure=DocumentFetchFailure(
                    reason="http_status",
                    status_code=403,
                    url="https://blocked.example/article",
                    message="private fetch failure",
                    source_error_type="HTTPStatusError",
                ),
            )
        ],
        extraction_outcomes=[],
        search_result_index_updates={},
        document_index_updates={},
    )
    return DefaultStateReducer().apply(
        AgentState(
            query=UserQuery(text="overall question"),
            scratch="private working notes",
        ),
        AgentAction(
            type=AgentActionType.RESEARCH,
            params={"query": observation.query, "route": observation.route.value},
        ),
        observation,
    )


def test_selects_from_context_and_candidate_summaries() -> None:
    client = FakeLLMClient('{"selected_refs":["C1"]}')
    selector = LLMSearchResultSelector(
        client,
        ResearchContextBuilder(InMemoryArtifactStore(), [RetrievalRoute.WEB]),
    )

    selection = asyncio.run(selector.select(_request(_state_with_fetch_failure())))

    assert selection.selected_ids == ["result-1"]
    llm_request = client.requests[0]
    assert llm_request.response_format == {"type": "json_object"}
    assert llm_request.metadata == {
        "trace": {"operation": "search_result_selector.select"}
    }
    prompt = json.loads(llm_request.messages[1].content)
    assert set(prompt["context"]) == {
        "user_query",
        "reference_time",
        "research_history",
        "evidence_groups",
    }
    assert "private working notes" not in llm_request.messages[1].content
    assert prompt["context"]["user_query"]["text"] == "overall question"
    assert prompt["context"]["research_history"] == [
        {
            "research_ref": "R1",
            "query": "prior query",
            "status": "completed",
            "fetch_failure_sources": [
                {
                    "domain": "blocked.example",
                    "reason": "http_status",
                    "status_code": 403,
                    "count": 1,
                }
            ],
        }
    ]
    assert prompt["current_search"] == {
        "query": "specific information gap",
        "candidate_results": [
            {
                "candidate_ref": "C1",
                "title": "Relevant result",
                "url": "https://example.com/relevant",
                "snippet": "Direct evidence",
                "rank": 1,
            },
            {
                "candidate_ref": "C2",
                "title": "Duplicate result",
                "url": "https://example.com/duplicate",
                "rank": 2,
            },
        ],
    }


@pytest.mark.parametrize(
    "output",
    ["not JSON", '{"selected_refs":["C3"]}'],
)
def test_rejects_invalid_llm_output(output: str) -> None:
    selector = LLMSearchResultSelector(
        FakeLLMClient(output),
        ResearchContextBuilder(InMemoryArtifactStore(), [RetrievalRoute.WEB]),
    )

    with pytest.raises(
        SearchResultSelectionError,
        match="invalid LLM search result selection",
    ):
        asyncio.run(selector.select(_request()))
