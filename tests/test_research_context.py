"""Tests for the evidence-oriented research context."""

from datetime import datetime, timezone

import pytest

from banso.artifacts.store import InMemoryArtifactStore
from banso.agent.action import (
    AgentAction,
    AgentActionType,
    RetrievalRoute,
)
from banso.agent.observation import (
    CompletedResearchObservation,
    CompletedSearchObservation,
    DocumentFetchFailure,
    EvidenceExtractionFailure,
    ExtractionFailure,
    ExtractionSuccess,
    FailedResearchObservation,
    FetchFailure,
    FetchSuccess,
    ReadObservation,
)
from banso.agent.reducer import DefaultStateReducer
from banso.agent.state import (
    ActionHistoryEntry,
    AgentState,
    ExecutionBudget,
    SearchResultState,
    UserQuery,
)
from banso.documents.models import Document, DocumentEvidence
from banso.agent.research_context import (
    CompletedQueryHistoryItem,
    ResearchContextBuilder,
)
from banso.retrieval.models import (
    RetrievalFilterReport,
    SearchResult,
    SearchResultMergeReport,
    SearchResultSelectionReport,
    SourceClassificationReport,
)
from banso.source import Source, SourceType


def _completed_state_and_store() -> tuple[AgentState, InMemoryArtifactStore]:
    store = InMemoryArtifactStore()
    source = Source(name="Publisher", type=SourceType.NEWS)
    result = SearchResult(
        id="result",
        title="Result",
        url="https://example.com/article",
        source=source,
    )
    document = Document(
        id="document",
        title="Document",
        url=result.url,
        text="Full body",
        source=source,
    )
    evidence = DocumentEvidence(
        id="evidence",
        document_id=document.id,
        text="Supported claim",
    )
    for artifact in (result, document, evidence):
        store.put(artifact)

    state = AgentState(
        query=UserQuery(text="question", language="en"),
        reference_time=datetime(2026, 8, 1, tzinfo=timezone.utc),
        budget=ExecutionBudget(max_researches=3),
    )
    action = AgentAction(
        type=AgentActionType.RESEARCH,
        params={
            "query": "focused query",
            "route": "web",
            "source_domains": ["example.com"],
        },
    )
    observation = CompletedResearchObservation(
        query="focused query",
        source_domains=["example.com"],
        search=CompletedSearchObservation(
            route=RetrievalRoute.WEB,
            search_result_ids=[result.id],
            search_result_index_updates={result.url: result.id},
            search_result_merge_report=SearchResultMergeReport(
                candidate_count=1,
                new_result_count=1,
                reused_result_count=0,
            ),
            retrieval_filter_report=RetrievalFilterReport(
                input_count=1,
                output_count=1,
            ),
            source_classification_report=SourceClassificationReport(
                input_count=1,
                recognized_count=1,
                unknown_count=0,
            ),
        ),
        selection_report=SearchResultSelectionReport(
            candidate_ids=[result.id],
            selected_ids=[result.id],
        ),
        read=ReadObservation(
            fetch_outcomes=[
                FetchSuccess(search_result_id=result.id, document_id=document.id)
            ],
            document_index_updates={document.url: document.id},
            extraction_outcomes=[
                ExtractionSuccess(document_id=document.id, evidence_id=evidence.id)
            ],
        ),
    )
    return DefaultStateReducer().apply(state, action, observation), store


def _research_entry(
    step_index: int,
    observation: CompletedResearchObservation | FailedResearchObservation,
) -> ActionHistoryEntry:
    return ActionHistoryEntry(
        step_index=step_index,
        action=AgentAction(
            type=AgentActionType.RESEARCH,
            params={
                "query": observation.query,
                "route": (
                    observation.route.value
                    if isinstance(observation, FailedResearchObservation)
                    else observation.search.route.value
                ),
            },
        ),
        observation=observation,
    )


def test_context_contains_research_history_and_evidence_groups() -> None:
    state, store = _completed_state_and_store()
    context = ResearchContextBuilder(
        store,
        [RetrievalRoute.LOCAL, RetrievalRoute.WEB],
    ).build(state)

    assert context.enabled_routes == [RetrievalRoute.LOCAL, RetrievalRoute.WEB]
    assert "language" not in context.user_query.model_dump()
    assert context.budget.remaining_researches == 2
    assert context.artifacts.search_result_count == 1
    assert context.artifacts.document_count == 1
    assert context.artifacts.evidence_document_count == 1
    history = context.retrieval_context.research_history
    evidence = context.evidence_context.evidence_groups
    assert len(history) == 1
    assert history[0].query_ref == "Q1"
    assert history[0].query == "focused query"
    assert history[0].source_domains == ["example.com"]
    assert history[0].selected_results == 1
    assert evidence[0].query_refs == ["Q1"]
    assert evidence[0].evidence_preview == "Supported claim"
    assert evidence[0].evidence_truncated is False
    dumped = context.model_dump(mode="json")
    assert dumped["evidence_context"]["notes"] == ""
    assert dumped["retrieval_context"]["candidate_results"] == []
    assert "candidate_documents" not in dumped
    assert "work" not in dumped


def test_completed_history_summarizes_evidence_and_fetch_failures() -> None:
    state, store = _completed_state_and_store()
    observation = state.action_history[0].observation
    assert isinstance(observation, CompletedResearchObservation)
    observation.read.fetch_outcomes.extend(
        [
            FetchFailure(
                search_result_id="failed-1",
                failure=DocumentFetchFailure(
                    reason="http_status",
                    status_code=403,
                    url="https://www.blocked.example/one",
                    message="private failure one",
                    source_error_type="HTTPStatusError",
                ),
            ),
            FetchFailure(
                search_result_id="failed-2",
                failure=DocumentFetchFailure(
                    reason="http_status",
                    status_code=403,
                    url="https://blocked.example/two",
                    message="private failure two",
                    source_error_type="HTTPStatusError",
                ),
            ),
        ]
    )
    observation.read.extraction_outcomes.extend(
        [
            ExtractionSuccess(document_id="empty-document"),
            ExtractionFailure(
                document_id="failed-document",
                failure=EvidenceExtractionFailure(
                    reason="invalid_response",
                    url="https://source.example/article",
                    message="private extraction failure",
                ),
            ),
        ]
    )

    context = ResearchContextBuilder(store, [RetrievalRoute.WEB]).build(state)

    history = context.retrieval_context.research_history[0]
    assert isinstance(history, CompletedQueryHistoryItem)
    assert history.fetch_failures == 2
    assert history.evidence_documents == 1
    assert history.no_evidence_documents == 1
    assert history.extraction_failures == 1
    assert [
        failure.model_dump(mode="json")
        for failure in history.fetch_failure_sources
    ] == [
        {
            "domain": "blocked.example",
            "reason": "http_status",
            "status_code": 403,
            "count": 2,
        }
    ]


def test_failed_research_still_advances_query_references() -> None:
    state, store = _completed_state_and_store()
    failure = FailedResearchObservation(
        query="failed query",
        route=RetrievalRoute.WEB,
        stage="retrieval",
        provider="test",
        reason="transport",
        message="failed",
        source_error_type="TransportError",
        retryable=True,
        attempt_count=1,
    )
    state.action_history.insert(0, _research_entry(0, failure))

    context = ResearchContextBuilder(store, [RetrievalRoute.WEB]).build(state)

    assert [
        item.query_ref for item in context.retrieval_context.research_history
    ] == [
        "Q1",
        "Q2",
    ]
    assert context.evidence_context.evidence_groups[0].query_refs == ["Q2"]


def test_unprocessed_result_keeps_the_research_that_returned_it() -> None:
    state, store = _completed_state_and_store()
    completed_entry = state.action_history[0]
    completed = completed_entry.observation
    assert isinstance(completed, CompletedResearchObservation)
    result_id = completed.search.search_result_ids[0]

    discovered = completed.model_copy(deep=True)
    discovered.query = "discovery query"
    discovered.selection_report = SearchResultSelectionReport(
        candidate_ids=[result_id],
        selected_ids=[],
    )
    discovered.read = ReadObservation(
        fetch_outcomes=[],
        extraction_outcomes=[],
        document_index_updates={},
    )

    fetched = completed.model_copy(deep=True)
    fetched.query = "later query"
    fetched.search.search_result_ids = []
    fetched.search.search_result_merge_report = SearchResultMergeReport(
        candidate_count=0,
        new_result_count=0,
        reused_result_count=0,
    )
    state.action_history = [
        _research_entry(0, discovered),
        _research_entry(1, fetched),
    ]

    context = ResearchContextBuilder(store, [RetrievalRoute.WEB]).build(state)

    assert [
        item.query for item in context.retrieval_context.research_history
    ] == [
        "discovery query",
        "later query",
    ]
    assert context.evidence_context.evidence_groups[0].query_refs == ["Q1"]


def test_document_query_references_are_ordered_unique_or_empty() -> None:
    state, store = _completed_state_and_store()
    first_entry = state.action_history[0]
    first = first_entry.observation
    assert isinstance(first, CompletedResearchObservation)
    state.search_results["redirected-result"] = SearchResultState(
        retrieval_route=RetrievalRoute.WEB,
        document_id="document",
    )
    first.search.search_result_ids.append("redirected-result")
    second = first.model_copy(deep=True)
    second.query = "another query"
    state.action_history.append(_research_entry(1, second))

    context = ResearchContextBuilder(store, [RetrievalRoute.WEB]).build(state)

    assert context.evidence_context.evidence_groups[0].query_refs == ["Q1", "Q2"]
    state.action_history = []
    context = ResearchContextBuilder(store, [RetrievalRoute.WEB]).build(state)
    assert context.evidence_context.evidence_groups[0].query_refs == []


def test_search_candidates_and_documents_share_query_references() -> None:
    state, store = _completed_state_and_store()
    existing_result_id = next(iter(state.search_results))
    candidate = SearchResult(
        id="candidate",
        title="Candidate",
        url="https://example.com/candidate",
    )
    store.put(candidate)
    search = AgentAction(
        type=AgentActionType.SEARCH,
        params={"query": "second query", "route": "web"},
    )
    state = DefaultStateReducer().apply(
        state,
        search,
        CompletedSearchObservation(
            route=RetrievalRoute.WEB,
            search_result_ids=[existing_result_id, candidate.id],
            retrieval_filter_report=RetrievalFilterReport(
                input_count=2,
                output_count=2,
            ),
            source_classification_report=SourceClassificationReport(
                input_count=2,
                recognized_count=0,
                unknown_count=2,
            ),
            search_result_merge_report=SearchResultMergeReport(
                candidate_count=2,
                new_result_count=1,
                reused_result_count=1,
            ),
            search_result_index_updates={candidate.url: candidate.id},
        ),
    )

    context = ResearchContextBuilder(store, [RetrievalRoute.WEB]).build(state)

    assert [
        (item.query_ref, item.query)
        for item in context.retrieval_context.research_history
    ] == [
        ("Q1", "focused query"),
        ("Q2", "second query"),
    ]
    assert context.retrieval_context.research_history[0].selected_results == 1
    assert context.retrieval_context.research_history[1].selected_results is None
    assert context.retrieval_context.candidate_results[0].candidate_ref == "C2"
    assert context.retrieval_context.candidate_results[0].query_refs == ["Q2"]
    assert context.evidence_context.evidence_groups[0].query_refs == ["Q1", "Q2"]


def test_context_limits_visible_evidence_text() -> None:
    state, store = _completed_state_and_store()

    context = ResearchContextBuilder(
        store,
        [RetrievalRoute.WEB],
        max_evidence_preview_chars=9,
    ).build(state)

    group = context.evidence_context.evidence_groups[0]
    assert group.evidence_preview == "Supported"
    assert group.evidence_truncated is True


def test_context_includes_all_evidence_documents_with_stable_refs() -> None:
    state, store = _completed_state_and_store()
    for index in range(1, 4):
        document = Document(
            id=f"document-{index}",
            title=f"Document {index}",
            url=f"https://example.com/{index}",
            text="body",
        )
        evidence = DocumentEvidence(
            id=f"evidence-{index}",
            document_id=document.id,
            text=f"Evidence {index}",
        )
        store.put(document)
        store.put(evidence)
        state.documents[document.id] = state.documents["document"].model_copy(
            update={"evidence_id": evidence.id}
        )

    context = ResearchContextBuilder(
        store,
        [RetrievalRoute.WEB],
    ).build(state)

    assert [
        group.document_ref for group in context.evidence_context.evidence_groups
    ] == [
        "D1",
        "D2",
        "D3",
        "D4",
    ]
    assert context.artifacts.evidence_document_count == 4


def test_context_rejects_missing_documents_and_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ResearchContextBuilder(InMemoryArtifactStore(), [])
    with pytest.raises(ValueError, match="unique"):
        ResearchContextBuilder(
            InMemoryArtifactStore(),
            [RetrievalRoute.WEB, RetrievalRoute.WEB],
        )
    state, store = _completed_state_and_store()
    state.documents["missing"] = state.documents["document"].model_copy(deep=True)
    with pytest.raises(ValueError, match="missing"):
        ResearchContextBuilder(store, [RetrievalRoute.WEB]).build(state)


def test_state_reference_time_defaults_to_utc() -> None:
    state = AgentState(query=UserQuery(text="query"))

    assert state.reference_time.tzinfo is not None
    assert state.reference_time.utcoffset().total_seconds() == 0
