"""Tests for compact news policy decision context."""

from datetime import datetime, timezone

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    ActionHistoryEntry,
    AgentActionType,
    AgentState,
    DocumentState,
    ExecutionBudget,
    ExtractProgress,
    Failure,
    Observation,
    SearchResultState,
    UserQuery,
)
from banso.documents import Document, EvidenceItem
from banso.policies import NewsPolicyContextBuilder
from banso.policies.news_policy_context import document_reference_maps
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
            url="https://www.example.com/result?tracking=1",
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


def _completed_state() -> AgentState:
    return AgentState(
        query=UserQuery(text="What happened?"),
        search_results={
            "result-1": SearchResultState(
                attempt_count=1,
                document_id="document-1",
            )
        },
        documents={
            "document-1": DocumentState(
                extraction=ExtractProgress(attempt_count=1),
                evidence_ids=["evidence-1"],
                lifecycle_status="active",
            ),
        },
    )


def test_state_initializes_reference_time_in_utc() -> None:
    state = AgentState(query=UserQuery(text="What happened?"))

    assert state.reference_time.tzinfo is timezone.utc
    assert state.reference_time.microsecond == 0


def test_builds_compact_context_from_completed_work() -> None:
    state = _completed_state()
    state.reference_time = datetime(2026, 7, 24, 8, 30, tzinfo=timezone.utc)
    state.current_step = 4
    state.budget = ExecutionBudget(
        max_steps=10,
        max_searches=3,
        max_document_fetches=6,
    )

    context = NewsPolicyContextBuilder(_populated_store()).build(state)

    assert context.user_query == UserQuery(text="What happened?")
    assert context.reference_time == datetime(
        2026,
        7,
        24,
        8,
        30,
        tzinfo=timezone.utc,
    )
    assert context.budget.model_dump() == {
        "remaining_steps": 6,
        "remaining_searches": 3,
        "remaining_document_fetches": 5,
        "max_active_documents": 6,
        "active_document_overflow": 0,
    }
    assert context.search_history == []
    assert context.work.model_dump() == {
        "fetch": {
            "pending": 0,
            "retryable": 0,
            "failed": 0,
            "actionable": 0,
            "failure_reasons": {},
        },
        "extraction": {
            "pending": 0,
            "retryable": 0,
            "failed": 0,
            "actionable": 0,
            "failure_reasons": {},
        },
        "extracted_without_evidence": 0,
    }
    assert context.artifacts.model_dump() == {
        "search_result_count": 1,
        "document_count": 1,
        "active_document_count": 1,
        "shelved_document_count": 0,
        "unusable_document_count": 0,
        "evidence_count": 1,
        "active_evidence_count": 1,
        "shelved_evidence_count": 0,
        "distinct_evidence_source_count": 1,
    }
    assert context.working_set.model_dump() == {
        "active_document_refs": ["D1"],
        "shelved_document_refs": [],
    }
    assert context.candidate_results == []
    assert context.candidate_documents == []
    assert [group.model_dump() for group in context.evidence_groups] == [
        {
            "document_ref": "D1",
            "lifecycle_status": "active",
            "lifecycle_reason": None,
            "lifecycle_updated_at_step": None,
            "document_title": "Document",
            "source": {
                "name": "Example",
                "domain": "example.com",
                "type": SourceType.NEWS,
            },
            "published_at": datetime(2026, 7, 17, tzinfo=timezone.utc),
            "evidence_count": 1,
            "claim_previews": ["evidence claim"],
        }
    ]

    serialized = context.model_dump_json()
    assert '"current_step"' not in serialized
    assert '"id"' not in serialized
    assert '"url"' not in serialized
    assert '"author"' not in serialized
    assert '"confidence"' not in serialized
    assert "supporting text must stay hidden" not in serialized


def test_builds_compact_search_history_without_raw_action_details() -> None:
    state = _completed_state()
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
            action_type=AgentActionType.FETCH_DOCUMENTS,
            observation=Observation(data={"private": "hidden fetch data"}),
        ),
    ]

    context = NewsPolicyContextBuilder(_populated_store()).build(state)

    assert context.budget.remaining_searches == 2
    assert [
        search.model_dump(mode="json") for search in context.search_history
    ] == [
        {
            "query": "AI launches",
            "intent": "find announcements",
            "new_results": 1,
            "reused_results": 1,
        }
    ]
    serialized = context.model_dump_json()
    assert "step_index" not in serialized
    assert "candidate_count" not in serialized
    assert "hidden fetch data" not in serialized


def test_exposes_only_actionable_resources_and_aggregates_failures() -> None:
    store = InMemoryArtifactStore()
    result_ids = [
        "result-retry",
        "result-succeeded",
        "result-pending",
        "result-failed",
    ]
    document_ids = [
        "document-retry",
        "document-empty",
        "document-pending",
        "document-failed",
    ]
    for artifact_id in result_ids:
        store.put(
            SearchResult(
                id=artifact_id,
                title=artifact_id,
                url=f"https://example.com/{artifact_id}",
            )
        )
    for artifact_id in document_ids:
        store.put(
            Document(
                id=artifact_id,
                title=artifact_id,
                url=f"https://example.com/{artifact_id}",
                text=artifact_id,
            )
        )

    state = AgentState(
        query=UserQuery(text="query"),
        search_results={
            "result-retry": SearchResultState(
                attempt_count=1,
                failure=Failure(reason="timeout", retryable=True),
            ),
            "result-succeeded": SearchResultState(
                attempt_count=1,
                document_id="document-empty",
            ),
            "result-pending": SearchResultState(),
            "result-failed": SearchResultState(
                attempt_count=2,
                failure=Failure(reason="timeout", retryable=True),
            ),
        },
        documents={
            "document-retry": DocumentState(
                extraction=ExtractProgress(
                    attempt_count=1,
                    failure=Failure(reason="llm_error", retryable=True),
                )
            ),
            "document-empty": DocumentState(
                extraction=ExtractProgress(attempt_count=1),
                lifecycle_status="unusable",
            ),
            "document-pending": DocumentState(),
            "document-failed": DocumentState(
                extraction=ExtractProgress(
                    attempt_count=1,
                    failure=Failure(reason="invalid_json", retryable=False),
                ),
                lifecycle_status="unusable",
            ),
        },
    )

    context = NewsPolicyContextBuilder(store).build(state)

    assert context.work.fetch.model_dump() == {
        "pending": 1,
        "retryable": 1,
        "failed": 1,
        "actionable": 2,
        "failure_reasons": {"timeout": 2},
    }
    assert context.work.extraction.model_dump() == {
        "pending": 1,
        "retryable": 1,
        "failed": 1,
        "actionable": 2,
        "failure_reasons": {
            "invalid_json": 1,
            "llm_error": 1,
        },
    }
    assert context.work.extracted_without_evidence == 1
    assert context.artifacts.unusable_document_count == 2
    assert [
        (result.title, result.fetch_status)
        for result in context.candidate_results
    ] == [
        ("result-pending", "pending"),
        ("result-retry", "retryable"),
    ]
    assert [
        (document.document_ref, document.title, document.extraction_status)
        for document in context.candidate_documents
    ] == [
        ("D3", "document-pending", "pending"),
        ("D1", "document-retry", "retryable"),
    ]
    assert [
        (
            group.document_ref,
            group.lifecycle_status,
            group.evidence_count,
        )
        for group in context.evidence_groups
    ] == [
        ("D2", "unusable", 0),
        ("D4", "unusable", 0),
    ]


def test_candidate_limits_do_not_change_actionable_or_artifact_counts() -> None:
    store = InMemoryArtifactStore()
    result_ids = ["result-2", "result-0", "result-1"]
    for artifact_id in result_ids:
        store.put(
            SearchResult(
                id=artifact_id,
                title=artifact_id,
                url=f"https://example.com/{artifact_id}",
            )
        )
    state = AgentState(
        query=UserQuery(text="query"),
        budget=ExecutionBudget(max_document_fetches=2),
        search_results={
            result_id: SearchResultState() for result_id in result_ids
        },
    )

    context = NewsPolicyContextBuilder(
        store,
        max_search_results=1,
    ).build(state)

    assert context.work.fetch.actionable == 2
    assert context.artifacts.search_result_count == 3
    assert [result.title for result in context.candidate_results] == [
        "result-2"
    ]


def test_limits_visible_evidence_per_document() -> None:
    store = InMemoryArtifactStore()
    sources = {
        "document-a": Source(
            name="Publisher A",
            type=SourceType.OFFICIAL,
        ),
        "document-b": Source(
            name="Publisher B",
            type=SourceType.NEWS,
        ),
    }
    for document_id, source in sources.items():
        store.put(
            Document(
                id=document_id,
                title=document_id,
                url=f"https://{document_id}.example/article",
                text=document_id,
                source=source,
            )
        )

    for evidence_id, document_id, source_url in (
        ("a-1", "document-a", "https://publisher-a.example/a-1"),
        ("a-2", "document-a", "https://publisher-a.example/a-2"),
        ("a-3", "document-a", "https://publisher-a.example/a-3"),
        ("b-1", "document-b", "https://publisher-b.example/b-1"),
        ("b-2", "document-b", "https://publisher-b.example/b-2"),
    ):
        store.put(
            EvidenceItem(
                id=evidence_id,
                document_id=document_id,
                claim=evidence_id,
                source_url=source_url,
            )
        )

    state = AgentState(
        query=UserQuery(text="query"),
        documents={
            "document-a": DocumentState(
                extraction=ExtractProgress(attempt_count=1),
                evidence_ids=["a-1", "a-2", "a-3"],
                lifecycle_status="active",
            ),
            "document-b": DocumentState(
                extraction=ExtractProgress(attempt_count=1),
                evidence_ids=["b-1", "b-2"],
                lifecycle_status="active",
            ),
        },
    )

    context = NewsPolicyContextBuilder(
        store,
        max_evidence_per_document=2,
    ).build(state)

    assert [
        group.claim_previews for group in context.evidence_groups
    ] == [["a-1", "a-2"], ["b-1", "b-2"]]
    assert [
        group.evidence_count for group in context.evidence_groups
    ] == [3, 2]
    assert context.artifacts.evidence_count == 5
    assert context.artifacts.distinct_evidence_source_count == 2
    assert [
        group.source.model_dump() for group in context.evidence_groups
    ] == [
        {
            "name": "Publisher A",
            "domain": "document-a.example",
            "type": SourceType.OFFICIAL,
        },
        {
            "name": "Publisher B",
            "domain": "document-b.example",
            "type": SourceType.NEWS,
        },
    ]

    hidden_context = NewsPolicyContextBuilder(store, max_evidence_per_document=0).build(state)
    assert [
        group.claim_previews for group in hidden_context.evidence_groups
    ] == [[], []]
    assert hidden_context.artifacts.evidence_count == 5


def test_truncates_visible_text_without_modifying_artifacts() -> None:
    store = _populated_store()
    store.put(
        Document(
            id="document-2",
            title="Pending document",
            url="https://example.com/document-2",
            text="document body",
        )
    )
    state = AgentState(
        query=UserQuery(text="query"),
        search_results={"result-1": SearchResultState()},
        documents={
            "document-1": DocumentState(
                extraction=ExtractProgress(attempt_count=1),
                evidence_ids=["evidence-1"],
                lifecycle_status="active",
            ),
            "document-2": DocumentState(),
        },
    )
    context = NewsPolicyContextBuilder(
        store,
        max_snippet_chars=6,
        max_document_preview_chars=8,
        max_claim_chars=8,
    ).build(state)

    assert context.candidate_results[0].snippet == "search"
    assert context.candidate_documents[0].text_preview == "document"
    assert context.evidence_groups[0].claim_previews == ["evidence"]
    stored_result = store.get("result-1", SearchResult)
    stored_document = store.get("document-1", Document)
    stored_evidence = store.get("evidence-1", EvidenceItem)
    assert stored_result is not None
    assert stored_document is not None
    assert stored_evidence is not None
    assert stored_result.snippet == "search snippet"
    assert stored_document.text == "document body"
    assert stored_evidence.claim == "evidence claim"


def test_document_references_are_stable_and_titles_need_not_be_unique() -> None:
    store = InMemoryArtifactStore()
    state = AgentState(query=UserQuery(text="query"))
    for document_id in ("document-a", "document-b"):
        store.put(
            Document(
                id=document_id,
                title="Repeated title",
                url=f"https://example.com/{document_id}",
                text="body",
            )
        )
        state.documents[document_id] = DocumentState(
            extraction=ExtractProgress(attempt_count=1),
            lifecycle_status="unusable",
        )

    initial_refs, _ = document_reference_maps(state)
    store.put(
        Document(
            id="document-c",
            title="Repeated title",
            url="https://example.com/document-c",
            text="body",
        )
    )
    state.documents["document-c"] = DocumentState()
    updated_refs, _ = document_reference_maps(state)
    context = NewsPolicyContextBuilder(store, max_documents=1).build(state)

    assert initial_refs == {
        "document-a": "D1",
        "document-b": "D2",
    }
    assert updated_refs == {
        "document-a": "D1",
        "document-b": "D2",
        "document-c": "D3",
    }
    assert [
        (group.document_ref, group.document_title)
        for group in context.evidence_groups
    ] == [
        ("D1", "Repeated title"),
        ("D2", "Repeated title"),
    ]


def test_shelved_evidence_group_uses_the_configured_claim_limit() -> None:
    store = InMemoryArtifactStore()
    store.put(
        Document(
            id="document-1",
            title="Document",
            url="https://example.com/document",
            text="body",
        )
    )
    evidence_ids: list[str] = []
    for index in range(3):
        evidence = EvidenceItem(
            id=f"evidence-{index}",
            document_id="document-1",
            claim=f"claim-{index}",
            source_url="https://example.com/document",
        )
        store.put(evidence)
        evidence_ids.append(evidence.id)
    state = AgentState(
        query=UserQuery(text="query"),
        documents={
            "document-1": DocumentState(
                extraction=ExtractProgress(attempt_count=1),
                evidence_ids=evidence_ids,
                lifecycle_status="shelved",
                lifecycle_reason="Duplicated stronger sources.",
                lifecycle_updated_at_step=4,
            )
        },
    )

    context = NewsPolicyContextBuilder(
        store,
        max_evidence_per_document=3,
    ).build(state)

    group = context.evidence_groups[0]
    assert group.claim_previews == ["claim-0", "claim-1", "claim-2"]
    assert group.lifecycle_reason == "Duplicated stronger sources."
    assert group.lifecycle_updated_at_step == 4
    assert context.working_set.model_dump() == {
        "active_document_refs": [],
        "shelved_document_refs": ["D1"],
    }
    assert context.artifacts.model_dump() == {
        "search_result_count": 0,
        "document_count": 1,
        "active_document_count": 0,
        "shelved_document_count": 1,
        "unusable_document_count": 0,
        "evidence_count": 3,
        "active_evidence_count": 0,
        "shelved_evidence_count": 3,
        "distinct_evidence_source_count": 1,
    }


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (
            AgentState(
                query=UserQuery(text="query"),
                search_results={"missing-result": SearchResultState()},
            ),
            "SearchResult artifact is missing or has the wrong type: missing-result",
        ),
        (
            AgentState(
                query=UserQuery(text="query"),
                documents={"result-1": DocumentState()},
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
        search_results={
            "result-1": SearchResultState(),
            "missing-result": SearchResultState(),
        },
    )

    with pytest.raises(ValueError, match="missing-result"):
        NewsPolicyContextBuilder(
            _populated_store(),
            max_search_results=1,
        ).build(state)


def test_context_and_inputs_are_isolated_snapshots() -> None:
    store = _populated_store()
    state = AgentState(
        query=UserQuery(text="original query"),
        search_results={"result-1": SearchResultState()},
    )
    context = NewsPolicyContextBuilder(store).build(state)

    state.query.text = "changed state"
    assert context.user_query.text == "original query"

    context.user_query.text = "changed context"
    context.candidate_results[0].source.name = "changed source"
    stored_result = store.get("result-1", SearchResult)
    assert stored_result is not None
    assert stored_result.source is not None
    assert stored_result.source.name == "Example"


def test_preserves_existing_default_limits() -> None:
    builder = NewsPolicyContextBuilder(_populated_store())

    assert builder.max_search_results == 30
    assert builder.max_documents == 8
    assert builder.max_evidence_per_document == 10
    assert builder.max_snippet_chars == 300
    assert builder.max_document_preview_chars == 750
    assert builder.max_claim_chars == 300


@pytest.mark.parametrize(
    "argument",
    [
        "max_search_results",
        "max_documents",
        "max_evidence_per_document",
        "max_snippet_chars",
        "max_document_preview_chars",
        "max_claim_chars",
    ],
)
def test_rejects_negative_limits(argument: str) -> None:
    with pytest.raises(ValueError, match=f"{argument} must be non-negative"):
        NewsPolicyContextBuilder(_populated_store(), **{argument: -1})
