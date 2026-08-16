"""Tests for the evidence-oriented news policy context."""

from datetime import datetime, timezone

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    AgentAction,
    AgentActionType,
    AgentState,
    DefaultStateReducer,
    ExecutionBudget,
    RetrievalRoute,
    UserQuery,
)
from banso.core.observation import (
    CompletedResearchObservation,
    ExtractionSuccess,
    FetchSuccess,
    RetrievalFilterReport,
    SearchResultMergeReport,
    SearchResultSelectionReport,
    SourceClassificationReport,
)
from banso.documents import Document, EvidenceItem
from banso.policies import NewsPolicyContextBuilder
from banso.retrieval import SearchResult, Source, SourceType


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
    evidence = EvidenceItem(
        id="evidence",
        document_id=document.id,
        claim="Supported claim",
        source_url=document.url,
    )
    for artifact in (result, document, evidence):
        store.put(artifact)

    state = AgentState(
        query=UserQuery(text="question", language="en"),
        reference_time=datetime(2026, 8, 1, tzinfo=timezone.utc),
        budget=ExecutionBudget(max_researches=3, max_document_fetches=6),
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
            deferred_ids=[],
        ),
        fetch_outcomes=[
            FetchSuccess(search_result_id=result.id, document_id=document.id)
        ],
        document_index_updates={document.url: document.id},
        extraction_outcomes=[
            ExtractionSuccess(document_id=document.id, evidence_ids=[evidence.id])
        ],
    )
    return DefaultStateReducer().apply(state, action, observation), store


def test_context_contains_research_history_and_evidence_groups() -> None:
    state, store = _completed_state_and_store()
    context = NewsPolicyContextBuilder(
        store,
        [RetrievalRoute.LOCAL, RetrievalRoute.WEB],
    ).build(state)

    assert context.enabled_routes == [RetrievalRoute.LOCAL, RetrievalRoute.WEB]
    assert "language" not in context.user_query.model_dump()
    assert context.budget.remaining_researches == 2
    assert context.budget.remaining_document_capacity == 5
    assert context.artifacts.search_result_count == 1
    assert context.artifacts.document_count == 1
    assert context.artifacts.active_evidence_count == 1
    assert context.working_set.active_document_refs == ["D1"]
    assert len(context.research_history) == 1
    assert context.research_history[0].query == "focused query"
    assert context.research_history[0].source_domains == ["example.com"]
    assert context.research_history[0].selected_results == 1
    assert context.evidence_groups[0].claim_previews == ["Supported claim"]
    dumped = context.model_dump(mode="json")
    assert "candidate_results" not in dumped
    assert "candidate_documents" not in dumped
    assert "work" not in dumped


def test_context_limits_visible_claims_without_changing_totals() -> None:
    state, store = _completed_state_and_store()
    second = EvidenceItem(
        id="evidence-2",
        document_id="document",
        claim="Second claim",
        source_url="https://example.com/article",
    )
    store.put(second)
    state.documents["document"].evidence_ids.append(second.id)

    context = NewsPolicyContextBuilder(
        store,
        [RetrievalRoute.WEB],
        max_evidence_per_document=1,
        max_claim_chars=9,
    ).build(state)

    assert context.artifacts.evidence_count == 2
    assert context.evidence_groups[0].evidence_count == 2
    assert context.evidence_groups[0].claim_previews == ["Supported"]


def test_document_references_are_stable_across_lifecycle_changes() -> None:
    state, store = _completed_state_and_store()
    builder = NewsPolicyContextBuilder(store, [RetrievalRoute.WEB])

    before = builder.build(state)
    state.documents["document"].lifecycle_status = "shelved"
    after = builder.build(state)

    assert before.evidence_groups[0].document_ref == "D1"
    assert after.evidence_groups[0].document_ref == "D1"
    assert after.working_set.shelved_document_refs == ["D1"]


def test_context_rejects_missing_artifacts_and_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="at least one"):
        NewsPolicyContextBuilder(InMemoryArtifactStore(), [])
    with pytest.raises(ValueError, match="unique"):
        NewsPolicyContextBuilder(
            InMemoryArtifactStore(),
            [RetrievalRoute.WEB, RetrievalRoute.WEB],
        )

    state, store = _completed_state_and_store()
    state.documents["missing"] = state.documents["document"].model_copy(deep=True)
    with pytest.raises(ValueError, match="missing"):
        NewsPolicyContextBuilder(store, [RetrievalRoute.WEB]).build(state)


def test_state_reference_time_defaults_to_utc() -> None:
    state = AgentState(query=UserQuery(text="query"))

    assert state.reference_time.tzinfo is not None
    assert state.reference_time.utcoffset().total_seconds() == 0
