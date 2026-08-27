"""Tests for news state derivations, validation, and failure classification."""

import pytest
from pydantic import ValidationError

from banso.agent.state import (
    AgentState,
    DocumentState,
    ExecutionBudget,
    Failure,
    SearchResultState,
    UserQuery,
)
from banso.documents.extractor import EvidenceExtractionError
from banso.documents.fetcher import DocumentFetchError


def test_state_derived_document_facts() -> None:
    state = AgentState(
        query=UserQuery(text="test"),
        search_results={
            "pending": SearchResultState(),
            "failed": SearchResultState(
                failure=Failure(reason="http_status"),
            ),
        },
        documents={
            "with-evidence": DocumentState(evidence_id="evidence"),
            "without-evidence": DocumentState(),
        },
    )

    assert state.evidence_document_count == 1
    assert state.has_evidence is True


@pytest.mark.parametrize(
    "budget",
    [
        {"max_researches": -1},
        {"max_results_per_research": 0},
    ],
)
def test_execution_budget_rejects_invalid_values(budget: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        ExecutionBudget(**budget)


def test_search_result_state_validates_pending_and_completed_outcomes() -> None:
    assert SearchResultState() == SearchResultState()
    assert SearchResultState(failure=Failure(reason="timeout")).failure is not None
    with pytest.raises(ValidationError, match="both"):
        SearchResultState(
            document_id="document",
            failure=Failure(reason="timeout"),
        )


@pytest.mark.parametrize(
    ("reason", "status_code", "expected"),
    [
        ("timeout", None, True),
        ("transport", None, True),
        ("http_status", 429, True),
        ("http_status", 500, True),
        ("http_status", 404, False),
        ("parse_error", None, False),
    ],
)
def test_document_fetch_failure_retryability(reason, status_code, expected) -> None:
    error = DocumentFetchError(
        url="https://example.com/article",
        reason=reason,
        message="failed",
        source_error_type="TestError",
        status_code=status_code,
    )
    assert error.retryable is expected


@pytest.mark.parametrize(
    ("reason", "expected"),
    [("llm_error", True), ("invalid_json", False), ("input_budget", False)],
)
def test_evidence_extraction_failure_retryability(reason: str, expected: bool) -> None:
    assert EvidenceExtractionError("failed", reason=reason).retryable is expected
