"""Tests for bounded news policy decision context."""

from datetime import datetime, timezone

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    ActionHistoryEntry,
    AgentActionType,
    AgentState,
    ExecutionBudget,
    ExtractProgress,
    Failure,
    Observation,
    ReadProgress,
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
        read_progress={
            "result-1": ReadProgress(
                attempt_count=1,
                document_id="document-1",
            )
        },
        extract_progress={
            "document-1": ExtractProgress(attempt_count=1),
        },
    )


def test_state_initializes_reference_time_in_utc() -> None:
    state = AgentState(query=UserQuery(text="What happened?"))

    assert state.reference_time.tzinfo is timezone.utc
    assert state.reference_time.microsecond == 0


def test_builds_context_with_budget_and_selected_artifact_fields() -> None:
    state = _state()
    state.reference_time = datetime(2026, 7, 24, 8, 30, tzinfo=timezone.utc)
    state.current_step = 4
    state.budget = ExecutionBudget(
        max_steps=10,
        max_searches=3,
        max_documents_to_read=6,
    )

    context = NewsPolicyContextBuilder(_populated_store()).build(state)

    assert context.user_query == UserQuery(text="What happened?")
    assert context.reference_time == datetime(2026, 7, 24, 8, 30, tzinfo=timezone.utc)
    assert context.current_step == 4
    assert context.max_steps == 10
    assert context.remaining_step_count == 6
    assert context.executed_search_count == 0
    assert context.max_searches == 3
    assert context.remaining_search_count == 3
    assert context.max_documents_to_read == 6
    assert context.remaining_document_count == 5
    assert context.searches == []
    assert context.search_result_count == 1
    assert context.omitted_search_result_count == 0
    assert context.pending_read_count == 0
    assert context.retryable_read_count == 0
    assert context.failed_read_count == 0
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
        "read_status": "succeeded",
        "document_id": "document-1",
        "read_failure_reason": None,
    }
    assert context.pending_extraction_count == 0
    assert context.retryable_extraction_count == 0
    assert context.failed_extraction_count == 0
    assert context.documents_without_evidence_count == 0
    assert context.documents[0].text_preview == "document body"
    assert context.documents[0].extraction_status == "succeeded"
    assert context.documents[0].evidence_count == 1
    assert "text" not in context.documents[0].model_dump()
    assert context.evidence[0].claim_preview == "evidence claim"
    assert "supporting_text" not in context.evidence[0].model_dump()


def test_builds_search_history_without_raw_action_details() -> None:
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
                    "read_outcomes": [
                        {
                            "search_result_id": "ignored-result",
                            "failure": {
                                "url": "https://example.com/ignored",
                                "reason": "timeout",
                                "retryable": True,
                                "message": "hidden failure detail",
                                "source_error_type": "ReadTimeout",
                            },
                        },
                    ],
                }
            ),
        ),
        ActionHistoryEntry(
            step_index=2,
            action_type=AgentActionType.EXTRACT_EVIDENCE,
            observation=Observation(
                data={
                    "extraction_outcomes": [
                        {
                            "document_id": "ignored-document",
                            "failure": {
                                "url": "https://example.com/ignored",
                                "reason": "invalid_json",
                                "retryable": False,
                                "message": "hidden parser response",
                            },
                        },
                    ],
                }
            ),
        ),
    ]

    context = NewsPolicyContextBuilder(_populated_store()).build(state)

    assert context.executed_search_count == 1
    assert context.remaining_search_count == 2
    assert [search.model_dump(mode="json") for search in context.searches] == [
        {
            "step_index": 0,
            "query": "AI launches",
            "intent": "find announcements",
            "result_count": 1,
            "new_result_count": 1,
            "reused_result_count": 1,
        }
    ]
    assert context.search_results[0].read_status == "succeeded"
    assert context.documents[0].extraction_status == "succeeded"
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

    assert context.searches == []


def test_prioritizes_actionable_resources_and_reports_lifecycle_counts() -> None:
    store = InMemoryArtifactStore()
    result_ids = ["result-pending", "result-succeeded", "result-retry", "result-failed"]
    document_ids = [
        "document-pending",
        "document-empty",
        "document-retry",
        "document-failed",
    ]
    for result_id in result_ids:
        store.put(
            SearchResult(
                id=result_id,
                title=result_id,
                url=f"https://example.com/{result_id}",
            )
        )
    for document_id in document_ids:
        store.put(
            Document(
                id=document_id,
                title=document_id,
                url=f"https://example.com/{document_id}",
                text=document_id,
            )
        )

    state = AgentState(
        query=UserQuery(text="query"),
        search_result_ids=result_ids,
        document_ids=document_ids,
        read_progress={
            "result-succeeded": ReadProgress(
                attempt_count=1,
                document_id="document-empty",
            ),
            "result-retry": ReadProgress(
                attempt_count=1,
                failure=Failure(reason="timeout", retryable=True),
            ),
            "result-failed": ReadProgress(
                attempt_count=2,
                failure=Failure(reason="timeout", retryable=True),
            ),
        },
        extract_progress={
            "document-empty": ExtractProgress(attempt_count=1),
            "document-retry": ExtractProgress(
                attempt_count=1,
                failure=Failure(reason="llm_error", retryable=True),
            ),
            "document-failed": ExtractProgress(
                attempt_count=1,
                failure=Failure(reason="invalid_json", retryable=False),
            ),
        },
    )

    context = NewsPolicyContextBuilder(
        store,
        max_search_results=2,
        max_documents=2,
    ).build(state)

    assert context.pending_read_count == 1
    assert context.retryable_read_count == 1
    assert context.failed_read_count == 1
    assert context.omitted_search_result_count == 2
    assert [result.id for result in context.search_results] == [
        "result-pending",
        "result-retry",
    ]
    assert context.search_results[1].read_failure_reason == "timeout"
    assert context.pending_extraction_count == 1
    assert context.retryable_extraction_count == 1
    assert context.failed_extraction_count == 1
    assert context.documents_without_evidence_count == 1
    assert context.omitted_document_count == 2
    assert [document.id for document in context.documents] == [
        "document-pending",
        "document-retry",
    ]
    assert context.documents[1].extraction_failure_reason == "llm_error"


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
