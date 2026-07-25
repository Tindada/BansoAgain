"""Tests for news resource lifecycle decisions."""

import pytest
from pydantic import ValidationError

from banso.core import (
    AgentState,
    ExecutionBudget,
    ExtractProgress,
    Failure,
    ReadProgress,
    UserQuery,
)
from banso.core.lifecycle import (
    eligible_extraction_document_ids,
    eligible_read_result_ids,
    extraction_status,
    read_status,
    remaining_document_count,
)
from banso.documents import DocumentReadError, EvidenceExtractionError


def test_read_lifecycle_prioritizes_pending_results_before_retries() -> None:
    state = AgentState(
        query=UserQuery(text="test"),
        budget=ExecutionBudget(max_documents_to_read=2, max_read_attempts=2),
        search_result_ids=["pending", "succeeded", "retryable", "failed"],
        document_ids=["document"],
        read_progress={
            "succeeded": ReadProgress(
                attempt_count=1,
                document_id="document",
            ),
            "retryable": ReadProgress(
                attempt_count=1,
                failure=Failure(reason="timeout", retryable=True),
            ),
            "failed": ReadProgress(
                attempt_count=1,
                failure=Failure(reason="http_status", retryable=False),
            ),
        },
    )

    assert read_status(state, "pending") == "pending"
    assert read_status(state, "succeeded") == "succeeded"
    assert read_status(state, "retryable") == "retryable"
    assert read_status(state, "failed") == "failed"
    assert remaining_document_count(state) == 1
    assert eligible_read_result_ids(state) == ["pending", "retryable"]


def test_read_lifecycle_exhausts_retries_and_document_budget() -> None:
    state = AgentState(
        query=UserQuery(text="test"),
        budget=ExecutionBudget(max_documents_to_read=1, max_read_attempts=2),
        search_result_ids=["exhausted"],
        read_progress={
            "exhausted": ReadProgress(
                attempt_count=2,
                failure=Failure(reason="timeout", retryable=True),
            )
        },
    )

    assert read_status(state, "exhausted") == "failed"
    assert eligible_read_result_ids(state) == []

    state.document_ids.append("document")
    assert remaining_document_count(state) == 0
    assert eligible_read_result_ids(state) == []


def test_extraction_lifecycle_distinguishes_empty_success_from_failures() -> None:
    state = AgentState(
        query=UserQuery(text="test"),
        budget=ExecutionBudget(max_extraction_attempts=2),
        document_ids=["pending", "empty", "retryable", "exhausted"],
        extract_progress={
            "empty": ExtractProgress(attempt_count=1),
            "retryable": ExtractProgress(
                attempt_count=1,
                failure=Failure(reason="llm_error", retryable=True),
            ),
            "exhausted": ExtractProgress(
                attempt_count=2,
                failure=Failure(reason="llm_error", retryable=True),
            ),
        },
    )

    assert extraction_status(state, "pending") == "pending"
    assert extraction_status(state, "empty") == "succeeded"
    assert extraction_status(state, "retryable") == "retryable"
    assert extraction_status(state, "exhausted") == "failed"
    assert eligible_extraction_document_ids(state) == ["pending", "retryable"]


def test_read_progress_requires_exactly_one_outcome() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        ReadProgress(attempt_count=1)

    with pytest.raises(ValidationError, match="exactly one"):
        ReadProgress(
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
