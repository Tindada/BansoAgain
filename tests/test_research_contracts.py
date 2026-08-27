"""Tests for the combined research action contracts."""

import pytest
from pydantic import ValidationError

from banso.agent.action import (
    AgentAction,
    AgentActionType,
    ResearchActionParams,
    RetrievalRoute,
)
from banso.agent.observation import (
    CompletedResearchObservation,
    DocumentFetchFailure,
    EvidenceExtractionFailure,
    ExtractionFailure,
    ExtractionSuccess,
    FetchFailure,
    FetchSuccess,
    RewriteScratchObservation,
)
from banso.agent.reducer import DefaultStateReducer
from banso.agent.state import AgentState, UserQuery
from banso.retrieval.models import (
    RetrievalFilterReport,
    SearchResultMergeReport,
    SearchResultSelectionReport,
    SourceClassificationReport,
)


def _research_observation() -> CompletedResearchObservation:
    return CompletedResearchObservation(
        query="query",
        route=RetrievalRoute.WEB,
        search_result_ids=["result-1"],
        search_result_index_updates={"https://example.com": "result-1"},
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
            recognized_count=0,
            unknown_count=1,
        ),
        selection_report=SearchResultSelectionReport(
            candidate_ids=["result-1"],
            selected_ids=["result-1"],
        ),
        fetch_outcomes=[
            FetchSuccess(
                search_result_id="result-1",
                document_id="document-1",
            )
        ],
        document_index_updates={"https://example.com": "document-1"},
        extraction_outcomes=[
            ExtractionSuccess(
                document_id="document-1",
                evidence_id="evidence-1",
            )
        ],
    )


def test_research_action_params_are_strict_and_normalize_query() -> None:
    params = ResearchActionParams(
        query="  query  ",
        route="web",
        source_domains=[" X.COM ", "news.example.com"],
    )

    assert params.query == "query"
    assert params.route == RetrievalRoute.WEB
    assert params.source_domains == ["x.com", "news.example.com"]
    assert ResearchActionParams(
        query="query", route="web", source_domains=[]
    ).source_domains is None

    with pytest.raises(ValidationError):
        ResearchActionParams(query=" ", route="web")
    with pytest.raises(ValidationError):
        ResearchActionParams(query="query", route="other")
    with pytest.raises(ValidationError):
        ResearchActionParams(query="query", route="web", unsupported=True)


@pytest.mark.parametrize(
    ("route", "source_domains"),
    [
        ("local", ["x.com"]),
        ("web", ["https://x.com"]),
        ("web", ["x.com/path"]),
        ("web", ["*.x.com"]),
        ("web", ["x.com", "X.COM"]),
    ],
)
def test_research_action_params_reject_invalid_source_domains(
    route: str,
    source_domains: list[str],
) -> None:
    with pytest.raises(ValidationError):
        ResearchActionParams(
            query="query",
            route=route,
            source_domains=source_domains,
        )


def test_reducer_replaces_scratch() -> None:
    state = AgentState(query=UserQuery(text="query"), scratch="old")
    action = AgentAction(type=AgentActionType.REWRITE_SCRATCH)

    next_state = DefaultStateReducer().apply(
        state,
        action,
        RewriteScratchObservation(content="new notes"),
    )

    assert state.scratch == "old"
    assert next_state.scratch == "new notes"
    assert next_state.action_history[0].action == action


def test_research_observation_reduces_the_entire_artifact_chain() -> None:
    state = AgentState(query=UserQuery(text="query"))
    action = AgentAction(
        type=AgentActionType.RESEARCH,
        params={"query": "query", "route": "web"},
    )

    next_state = DefaultStateReducer().apply(
        state,
        action,
        _research_observation(),
    )

    assert next_state.search_results["result-1"].document_id == "document-1"
    assert next_state.documents["document-1"].evidence_id == "evidence-1"
    assert next_state.action_history[0].observation.type == AgentActionType.RESEARCH


def test_reducer_records_terminal_fetch_failure() -> None:
    observation = _research_observation().model_copy(
        update={
            "fetch_outcomes": [
                FetchFailure(
                    search_result_id="result-1",
                    failure=DocumentFetchFailure(
                        reason="timeout",
                        url="https://example.com",
                        message="timed out",
                        source_error_type="TimeoutError",
                    ),
                    attempt_count=2,
                )
            ],
            "document_index_updates": {},
            "extraction_outcomes": [],
        }
    )

    next_state = DefaultStateReducer().apply(
        AgentState(query=UserQuery(text="query")),
        AgentAction(
            type=AgentActionType.RESEARCH,
            params={"query": "query", "route": "web"},
        ),
        observation,
    )

    assert next_state.search_results["result-1"].failure.reason == "timeout"
    assert next_state.documents == {}


def test_reducer_records_document_without_evidence_after_extraction_failure() -> None:
    observation = _research_observation().model_copy(
        update={
            "extraction_outcomes": [
                ExtractionFailure(
                    document_id="document-1",
                    failure=EvidenceExtractionFailure(
                        reason="llm_error",
                        url="https://example.com",
                        message="provider failed",
                    ),
                    attempt_count=2,
                )
            ]
        }
    )

    next_state = DefaultStateReducer().apply(
        AgentState(query=UserQuery(text="query")),
        AgentAction(
            type=AgentActionType.RESEARCH,
            params={"query": "query", "route": "web"},
        ),
        observation,
    )

    document = next_state.documents["document-1"]
    assert document.evidence_id is None
    recorded = next_state.action_history[0].observation
    assert isinstance(recorded, CompletedResearchObservation)
    outcome = recorded.extraction_outcomes[0]
    assert isinstance(outcome, ExtractionFailure)
    assert outcome.failure.reason == "llm_error"


def test_reducer_records_document_without_evidence_after_empty_extraction() -> None:
    observation = _research_observation().model_copy(
        update={
            "extraction_outcomes": [
                ExtractionSuccess(document_id="document-1")
            ]
        }
    )

    next_state = DefaultStateReducer().apply(
        AgentState(query=UserQuery(text="query")),
        AgentAction(
            type=AgentActionType.RESEARCH,
            params={"query": "query", "route": "web"},
        ),
        observation,
    )

    document = next_state.documents["document-1"]
    assert document.evidence_id is None
    recorded = next_state.action_history[0].observation
    assert isinstance(recorded, CompletedResearchObservation)
    outcome = recorded.extraction_outcomes[0]
    assert isinstance(outcome, ExtractionSuccess)
    assert outcome.evidence_id is None
