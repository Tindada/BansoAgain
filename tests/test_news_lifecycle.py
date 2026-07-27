"""Tests for news resource lifecycle decisions."""

import pytest
from pydantic import ValidationError

from banso.core import (
    AgentState,
    DocumentState,
    ExecutionBudget,
    ExtractProgress,
    Failure,
    SearchResultState,
    UserQuery,
)
from banso.core.lifecycle import (
    eligible_extraction_document_ids,
    eligible_read_result_ids,
    progress_status,
    remaining_document_count,
)
from banso.documents import DocumentReadError, EvidenceExtractionError


@pytest.mark.parametrize(
    ("progress", "max_attempts", "expected"),
    [
        (None, 2, "pending"),
        (SearchResultState(), 2, "pending"),
        (
            SearchResultState(attempt_count=1, document_id="document"),
            2,
            "succeeded",
        ),
        (ExtractProgress(attempt_count=1), 2, "succeeded"),
        (
            SearchResultState(
                attempt_count=1,
                failure=Failure(reason="timeout", retryable=True),
            ),
            2,
            "retryable",
        ),
        (
            ExtractProgress(
                attempt_count=2,
                failure=Failure(reason="llm_error", retryable=True),
            ),
            2,
            "failed",
        ),
        (
            SearchResultState(
                attempt_count=1,
                failure=Failure(reason="http_status", retryable=False),
            ),
            2,
            "failed",
        ),
    ],
)
def test_progress_status(
    progress: SearchResultState | ExtractProgress | None,
    max_attempts: int,
    expected: str,
) -> None:
    assert progress_status(progress, max_attempts) == expected


def test_read_lifecycle_prioritizes_pending_results_before_retries() -> None:
    state = AgentState(
        query=UserQuery(text="test"),
        budget=ExecutionBudget(max_documents_to_read=2, max_read_attempts=2),
        search_results={
            "pending": SearchResultState(),
            "succeeded": SearchResultState(
                attempt_count=1,
                document_id="document",
            ),
            "retryable": SearchResultState(
                attempt_count=1,
                failure=Failure(reason="timeout", retryable=True),
            ),
            "failed": SearchResultState(
                attempt_count=1,
                failure=Failure(reason="http_status", retryable=False),
            ),
        },
        documents={"document": DocumentState()},
    )

    assert remaining_document_count(state) == 1
    assert eligible_read_result_ids(state) == ["pending", "retryable"]


def test_read_lifecycle_exhausts_retries_and_document_budget() -> None:
    state = AgentState(
        query=UserQuery(text="test"),
        budget=ExecutionBudget(max_documents_to_read=1, max_read_attempts=2),
        search_results={
            "exhausted": SearchResultState(
                attempt_count=2,
                failure=Failure(reason="timeout", retryable=True),
            ),
            "pending": SearchResultState(),
        },
    )

    assert eligible_read_result_ids(state) == ["pending"]

    state.documents["document"] = DocumentState()
    assert remaining_document_count(state) == 0
    assert eligible_read_result_ids(state) == []


def test_extraction_lifecycle_distinguishes_empty_success_from_failures() -> None:
    state = AgentState(
        query=UserQuery(text="test"),
        budget=ExecutionBudget(max_extraction_attempts=2),
        documents={
            "pending": DocumentState(),
            "empty": DocumentState(
                extraction=ExtractProgress(attempt_count=1)
            ),
            "retryable": DocumentState(
                extraction=ExtractProgress(
                    attempt_count=1,
                    failure=Failure(reason="llm_error", retryable=True),
                )
            ),
            "exhausted": DocumentState(
                extraction=ExtractProgress(
                    attempt_count=2,
                    failure=Failure(reason="llm_error", retryable=True),
                )
            ),
        },
    )

    assert eligible_extraction_document_ids(state) == ["pending", "retryable"]


def test_search_result_state_validates_pending_and_completed_outcomes() -> None:
    with pytest.raises(ValidationError, match="pending"):
        SearchResultState(
            failure=Failure(reason="timeout", retryable=True),
        )

    with pytest.raises(ValidationError, match="exactly one"):
        SearchResultState(attempt_count=1)

    with pytest.raises(ValidationError, match="exactly one"):
        SearchResultState(
            attempt_count=1,
            document_id="document",
            failure=Failure(reason="timeout", retryable=True),
        )


@pytest.mark.parametrize(
    ("reason", "status_code", "expected"),
    [
        ("timeout", None, True),
        ("transport", None, True),
        ("http_status", 408, True),
        ("http_status", 425, True),
        ("http_status", 429, True),
        ("http_status", 500, True),
        ("http_status", 599, True),
        ("http_status", 404, False),
        ("http_status", 600, False),
        ("parse_error", None, False),
    ],
)
def test_document_read_failure_retryability(
    reason,
    status_code: int | None,
    expected: bool,
) -> None:
    error = DocumentReadError(
        url="https://example.com/article",
        reason=reason,
        message="failed",
        source_error_type="TestError",
        status_code=status_code,
    )

    assert error.retryable is expected


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("llm_error", True),
        ("invalid_json", False),
        ("invalid_schema", False),
        ("input_budget", False),
        ("document_too_large", False),
    ],
)
def test_evidence_extraction_failure_retryability(
    reason: str,
    expected: bool,
) -> None:
    error = EvidenceExtractionError("failed", reason=reason)

    assert error.retryable is expected
