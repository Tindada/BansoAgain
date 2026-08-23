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
    ExtractionSuccess,
    FetchSuccess,
    RetrievalFailedResearchObservation,
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
from banso.agent.research_context import ResearchContextBuilder
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
        budget=ExecutionBudget(max_researches=3, max_active_documents=6),
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
        route=RetrievalRoute.WEB,
        source_domains=["example.com"],
        search_result_ids=[result.id],
        search_result_index_updates={result.url: result.id},
        search_result_merge_report=SearchResultMergeReport(
            candidate_count=1,
            new_result_count=1,
            reused_result_count=0,
        ),
        retrieval_filter_report=RetrievalFilterReport(input_count=1, output_count=1),
        source_classification_report=SourceClassificationReport(
            input_count=1,
            recognized_count=1,
            unknown_count=0,
        ),
        selection_report=SearchResultSelectionReport(
            candidate_ids=[result.id],
            selected_ids=[result.id],
        ),
        fetch_outcomes=[
            FetchSuccess(search_result_id=result.id, document_id=document.id)
        ],
        document_index_updates={document.url: document.id},
        extraction_outcomes=[
            ExtractionSuccess(document_id=document.id, evidence_id=evidence.id)
        ],
    )
    return DefaultStateReducer().apply(state, action, observation), store


def _research_entry(
    step_index: int,
    observation: CompletedResearchObservation | RetrievalFailedResearchObservation,
) -> ActionHistoryEntry:
    return ActionHistoryEntry(
        step_index=step_index,
        action=AgentAction(
            type=AgentActionType.RESEARCH,
            params={"query": observation.query, "route": observation.route.value},
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
    assert context.artifacts.active_document_count == 1
    assert context.working_set.active_document_refs == ["D1"]
    assert len(context.research_history) == 1
    assert context.research_history[0].research_ref == "R1"
    assert context.research_history[0].query == "focused query"
    assert context.research_history[0].source_domains == ["example.com"]
    assert context.research_history[0].selected_results == 1
    assert context.evidence_groups[0].research_refs == ["R1"]
    assert context.evidence_groups[0].evidence_preview == "Supported claim"
    assert context.evidence_groups[0].evidence_truncated is False
    dumped = context.model_dump(mode="json")
    assert "candidate_results" not in dumped
    assert "candidate_documents" not in dumped
    assert "work" not in dumped


def test_failed_research_still_advances_research_references() -> None:
    state, store = _completed_state_and_store()
    failure = RetrievalFailedResearchObservation(
        query="failed query",
        route=RetrievalRoute.WEB,
        provider="test",
        reason="transport",
        message="failed",
        source_error_type="TransportError",
        retryable=True,
        attempt_count=1,
    )
    state.action_history.insert(0, _research_entry(0, failure))

    context = ResearchContextBuilder(store, [RetrievalRoute.WEB]).build(state)

    assert [item.research_ref for item in context.research_history] == [
        "R1",
        "R2",
    ]
    assert context.evidence_groups[0].research_refs == ["R2"]


def test_unprocessed_result_keeps_the_research_that_returned_it() -> None:
    state, store = _completed_state_and_store()
    completed_entry = state.action_history[0]
    completed = completed_entry.observation
    assert isinstance(completed, CompletedResearchObservation)
    result_id = completed.search_result_ids[0]

    discovered = completed.model_copy(deep=True)
    discovered.query = "discovery query"
    discovered.selection_report = SearchResultSelectionReport(
        candidate_ids=[result_id],
        selected_ids=[],
    )
    discovered.fetch_outcomes = []
    discovered.extraction_outcomes = []
    discovered.document_index_updates = {}

    fetched = completed.model_copy(deep=True)
    fetched.query = "later query"
    fetched.search_result_ids = []
    fetched.search_result_merge_report = SearchResultMergeReport(
        candidate_count=0,
        new_result_count=0,
        reused_result_count=0,
    )
    state.action_history = [
        _research_entry(0, discovered),
        _research_entry(1, fetched),
    ]

    context = ResearchContextBuilder(store, [RetrievalRoute.WEB]).build(state)

    assert [item.query for item in context.research_history] == [
        "discovery query",
        "later query",
    ]
    assert context.evidence_groups[0].research_refs == ["R1"]


def test_document_research_references_are_ordered_unique_or_empty() -> None:
    state, store = _completed_state_and_store()
    first_entry = state.action_history[0]
    first = first_entry.observation
    assert isinstance(first, CompletedResearchObservation)
    state.search_results["redirected-result"] = SearchResultState(
        document_id="document"
    )
    first.search_result_ids.append("redirected-result")
    second = first.model_copy(deep=True)
    second.query = "another query"
    state.action_history.append(_research_entry(1, second))

    context = ResearchContextBuilder(store, [RetrievalRoute.WEB]).build(state)

    assert context.evidence_groups[0].research_refs == ["R1", "R2"]
    state.action_history = []
    context = ResearchContextBuilder(store, [RetrievalRoute.WEB]).build(state)
    assert context.evidence_groups[0].research_refs == []


def test_context_limits_visible_evidence_text() -> None:
    state, store = _completed_state_and_store()

    context = ResearchContextBuilder(
        store,
        [RetrievalRoute.WEB],
        max_evidence_preview_chars=9,
    ).build(state)

    group = context.evidence_groups[0]
    assert group.evidence_preview == "Supported"
    assert group.evidence_truncated is True


def test_document_references_are_stable_across_lifecycle_changes() -> None:
    state, store = _completed_state_and_store()
    builder = ResearchContextBuilder(store, [RetrievalRoute.WEB])

    before = builder.build(state)
    state.documents["document"].lifecycle_status = "shelved"
    after = builder.build(state)

    assert before.evidence_groups[0].document_ref == "D1"
    assert after.evidence_groups[0].document_ref == "D1"
    assert after.working_set.shelved_document_refs == ["D1"]


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
