"""Tests for LLM-backed search-result selection."""

import asyncio
import json

import pytest

from banso.agent.action import RetrievalRoute
from banso.agent.research_context import ResearchContextBuilder
from banso.agent.selection.llm_selector import LLMSearchResultSelector
from banso.agent.selection.selector import SearchResultSelectionRequest
from banso.agent.state import AgentState, UserQuery
from banso.artifacts.store import InMemoryArtifactStore
from banso.llm.fake import FakeLLMClient
from banso.retrieval.models import SearchResult


def _request() -> SearchResultSelectionRequest:
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
        state=AgentState(query=UserQuery(text="overall question")),
    )


def test_selects_from_context_and_candidate_summaries() -> None:
    client = FakeLLMClient('{"selected_refs":["C1"]}')
    selector = LLMSearchResultSelector(
        client,
        ResearchContextBuilder(InMemoryArtifactStore(), [RetrievalRoute.WEB]),
    )

    selection = asyncio.run(selector.select(_request()))

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
    assert prompt["context"]["user_query"]["text"] == "overall question"
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

    with pytest.raises(ValueError, match="invalid LLM search result selection"):
        asyncio.run(selector.select(_request()))
