"""Tests for bounded news policy decision context."""

from datetime import datetime, timezone

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    ActionHistoryEntry,
    AgentActionType,
    AgentState,
    ExecutionBudget,
    Observation,
    UserQuery,
)
from banso.documents import Document, EvidenceItem
from banso.policies import NewsPolicyContextBuilder
from banso.retrieval import SearchResult, Source, SourceType


def _populated_store() -> InMemoryArtifactStore:
    store = InMemoryArtifactStore()
    source = Source(
        name="Example",
        url="https://example.com",
        type=SourceType.NEWS,
    )
    published_at = datetime(2026, 7, 17, tzinfo=timezone.utc)
    store.put(
        SearchResult(
            id="result-1",
            title="Result",
            url="https://example.com/result",
            snippet="search snippet",
            source=source,
            published_at=published_at,
            metadata={"provider_payload": "hidden"},
        )
    )
    store.put(
        Document(
            id="document-1",
            title="Document",
            url="https://example.com/document",
            text="document body",
            source=source,
            published_at=published_at,
            author="Author",
            metadata={"parser_output": "hidden"},
        )
    )
    store.put(
        EvidenceItem(
            id="evidence-1",
            document_id="document-1",
            claim="evidence claim",
            supporting_text="supporting text must stay hidden",
            source_url="https://example.com/document",
            published_at=published_at,
            confidence=0.9,
            metadata={"extractor_output": "hidden"},
        )
    )
    return store


def _state() -> AgentState:
    return AgentState(
        query=UserQuery(text="What happened?"),
        search_result_ids=["result-1"],
        document_ids=["document-1"],
        evidence_ids=["evidence-1"],
    )


def test_builds_context_with_budget_and_selected_artifact_fields() -> None:
    state = _state()
    state.current_step = 4
    state.budget = ExecutionBudget(
        max_steps=10,
        max_searches=3,
        max_documents_to_read=6,
    )

    context = NewsPolicyContextBuilder(_populated_store()).build(state)

    assert context.user_query == UserQuery(text="What happened?")
    assert context.current_step == 4
    assert context.max_steps == 10
    assert context.remaining_step_count == 6
    assert context.executed_search_count == 0
    assert context.max_searches == 3
    assert context.remaining_search_count == 3
    assert context.max_documents_to_read == 6
    assert context.search_result_count == 1
    assert context.omitted_search_result_count == 0
    assert len(context.search_results) == 1
    assert context.search_results[0].model_dump() == {
        "id": "result-1",
        "title": "Result",
        "url": "https://example.com/result",
        "snippet": "search snippet",
        "source": {
            "name": "Example",
            "url": "https://example.com",
            "type": SourceType.NEWS,
        },
        "published_at": datetime(2026, 7, 17, tzinfo=timezone.utc),
    }
    assert context.documents[0].text_preview == "document body"
    assert "text" not in context.documents[0].model_dump()
    assert context.evidence[0].claim_preview == "evidence claim"
    assert "supporting_text" not in context.evidence[0].model_dump()


def test_builds_typed_attempts_without_raw_observation_details() -> None:
    state = _state()
    state.current_step = 3
    state.action_history = [
        ActionHistoryEntry(
            step_index=0,
            action_type=AgentActionType.SEARCH,
            params={"query": "AI launches", "intent": "find announcements"},
            observation=Observation(
                data={
                    "search_result_ids": ["result-1"],
                    "search_result_merge_report": {
                        "candidate_count": 2,
                        "new_result_count": 1,
                        "reused_result_count": 1,
                    },
                    "retrieval_filter_report": {"input_count": 99},
                }
            ),
        ),
        ActionHistoryEntry(
            step_index=1,
            action_type=AgentActionType.READ_DOCUMENT,
            observation=Observation(
                data={
                    "document_ids": ["document-1"],
                    "document_read_failures": [
                        {
                            "search_result_id": "result-2",
                            "url": "https://example.com/failed",
                            "status_code": 503,
                            "reason": "http_status",
                            "message": "hidden failure detail",
                            "source_error_type": "HTTPStatusError",
                        }
                    ],
                }
            ),
        ),
        ActionHistoryEntry(
            step_index=2,
            action_type=AgentActionType.EXTRACT_EVIDENCE,
            observation=Observation(
                data={
                    "evidence_ids": ["evidence-1"],
                    "successful_document_count": 1,
                    "documents_without_evidence": ["document-2"],
                    "evidence_extraction_failures": [
                        {
                            "document_id": "document-3",
                            "url": "https://example.com/failed-extraction",
                            "reason": "invalid_json",
                            "message": "hidden parser response",
                        }
                    ],
                }
            ),
        ),
    ]

    context = NewsPolicyContextBuilder(_populated_store()).build(state)

    assert context.executed_search_count == 1
    assert context.remaining_search_count == 2
    assert [attempt.model_dump(mode="json") for attempt in context.attempts] == [
        {
            "type": "search",
            "step_index": 0,
            "query": "AI launches",
            "intent": "find announcements",
            "result_ids": ["result-1"],
            "new_result_count": 1,
            "reused_result_count": 1,
        },
        {
            "type": "read_document",
            "step_index": 1,
            "document_ids": ["document-1"],
            "failures": [
                {
                    "resource_id": "result-2",
                    "url": "https://example.com/failed",
                    "reason": "http_status",
                    "status_code": 503,
                }
            ],
        },
        {
            "type": "extract_evidence",
            "step_index": 2,
            "evidence_ids": ["evidence-1"],
            "successful_document_count": 1,
            "documents_without_evidence": ["document-2"],
            "failures": [
                {
                    "resource_id": "document-3",
                    "url": "https://example.com/failed-extraction",
                    "reason": "invalid_json",
                    "status_code": None,
                }
            ],
        },
    ]
    serialized = context.model_dump_json()
    assert "retrieval_filter_report" not in serialized
    assert "hidden failure detail" not in serialized
    assert "hidden parser response" not in serialized


def test_ignores_actions_that_do_not_inform_a_future_llm_decision() -> None:
    state = _state()
    state.action_history = [
        ActionHistoryEntry(
            step_index=0,
            action_type=AgentActionType.PLAN_SEARCH,
            observation=Observation(data={"search_plan": {"searches": []}}),
        ),
        ActionHistoryEntry(
            step_index=1,
            action_type=AgentActionType.FINISH,
            observation=Observation(data={"final_answer": "answer"}),
        ),
    ]

    context = NewsPolicyContextBuilder(_populated_store()).build(state)

    assert context.attempts == []


def test_preserves_state_order_and_reports_item_limits() -> None:
    store = InMemoryArtifactStore()
    for index in range(3):
        store.put(
            SearchResult(
                id=f"result-{index}",
                title=f"Result {index}",
                url=f"https://example.com/{index}",
            )
        )
    state = AgentState(
        query=UserQuery(text="query"),
        search_result_ids=["result-2", "result-0", "result-1"],
    )

    context = NewsPolicyContextBuilder(
        store,
        max_search_results=2,
    ).build(state)

    assert context.search_result_count == 3
    assert context.omitted_search_result_count == 1
    assert [result.id for result in context.search_results] == [
        "result-2",
        "result-0",
    ]


def test_truncates_visible_text_without_modifying_artifacts() -> None:
    store = _populated_store()
    context = NewsPolicyContextBuilder(
        store,
        max_snippet_chars=6,
        max_document_preview_chars=8,
        max_claim_chars=8,
    ).build(_state())

    assert context.search_results[0].snippet == "search"
    assert context.documents[0].text_preview == "document"
    assert context.evidence[0].claim_preview == "evidence"
    stored_result = store.get("result-1", SearchResult)
    stored_document = store.get("document-1", Document)
    stored_evidence = store.get("evidence-1", EvidenceItem)
    assert stored_result is not None
    assert stored_document is not None
    assert stored_evidence is not None
    assert stored_result.snippet == "search snippet"
    assert stored_document.text == "document body"
    assert stored_evidence.claim == "evidence claim"


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (
            AgentState(
                query=UserQuery(text="query"),
                search_result_ids=["missing-result"],
            ),
            "SearchResult artifact is missing or has the wrong type: missing-result",
        ),
        (
            AgentState(
                query=UserQuery(text="query"),
                document_ids=["result-1"],
            ),
            "Document artifact is missing or has the wrong type: result-1",
        ),
    ],
)
def test_rejects_missing_or_wrongly_typed_artifact(
    state: AgentState,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        NewsPolicyContextBuilder(_populated_store()).build(state)


def test_validates_artifact_ids_beyond_visible_limit() -> None:
    state = AgentState(
        query=UserQuery(text="query"),
        search_result_ids=["result-1", "missing-result"],
    )

    with pytest.raises(ValueError, match="missing-result"):
        NewsPolicyContextBuilder(
            _populated_store(),
            max_search_results=1,
        ).build(state)


def test_context_and_inputs_are_isolated_snapshots() -> None:
    store = _populated_store()
    state = _state()
    context = NewsPolicyContextBuilder(store).build(state)

    state.query.text = "changed state"
    assert context.user_query.text == "What happened?"

    context.user_query.text = "changed context"
    assert state.query.text == "changed state"
    assert context.user_query.text == "changed context"

    context_source = context.search_results[0].source
    assert context_source is not None
    context_source.name = "changed source"
    stored_result = store.get("result-1", SearchResult)
    assert stored_result is not None
    assert stored_result.source is not None
    assert stored_result.source.name == "Example"


@pytest.mark.parametrize(
    "argument",
    [
        "max_search_results",
        "max_documents",
        "max_evidence",
        "max_snippet_chars",
        "max_document_preview_chars",
        "max_claim_chars",
    ],
)
def test_rejects_negative_limits(argument: str) -> None:
    with pytest.raises(ValueError, match=f"{argument} must be non-negative"):
        NewsPolicyContextBuilder(_populated_store(), **{argument: -1})
