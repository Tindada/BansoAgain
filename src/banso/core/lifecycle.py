"""Shared resource lifecycle decisions."""

from typing import Literal

from banso.core.state import AgentState, ExtractProgress, SearchResultState

LifecycleStatus = Literal["pending", "succeeded", "retryable", "failed"]


def progress_status(
    progress: SearchResultState | ExtractProgress | None,
    max_attempts: int,
) -> LifecycleStatus:
    """Derive one processing status from its progress and attempt limit."""
    if progress is None or progress.attempt_count == 0:
        return "pending"
    if progress.failure is None:
        return "succeeded"
    if progress.failure.retryable and progress.attempt_count < max_attempts:
        return "retryable"
    return "failed"


def active_document_count(state: AgentState) -> int:
    """Return the number of documents in the active working set."""
    return sum(document.lifecycle_status == "active" for document in state.documents.values())


def remaining_document_fetches(state: AgentState) -> int:
    """Return the number of unique documents the run may still collect."""
    return max(state.budget.max_document_fetches - len(state.documents), 0)


def eligible_fetch_result_ids(state: AgentState) -> list[str]:
    """Return unfetched results followed by retryable fetch failures."""
    pending: list[str] = []
    retryable: list[str] = []
    for result_id, result in state.search_results.items():
        status = progress_status(result, state.budget.max_fetch_attempts)
        if status == "pending":
            pending.append(result_id)
        elif status == "retryable":
            retryable.append(result_id)
    return [*pending, *retryable]


def eligible_extraction_document_ids(state: AgentState) -> list[str]:
    """Return unprocessed documents followed by retryable extraction failures."""
    pending: list[str] = []
    retryable: list[str] = []
    for document_id, document in state.documents.items():
        if document.lifecycle_status is not None:
            continue
        status = progress_status(document.extraction, state.budget.max_extraction_attempts)
        if status == "pending":
            pending.append(document_id)
        elif status == "retryable":
            retryable.append(document_id)
    return [*pending, *retryable]


def curatable_document_ids(state: AgentState) -> list[str]:
    """Return evidence-bearing documents available for agent curation."""
    return [
        document_id
        for document_id, document in state.documents.items()
        if document.lifecycle_status in {"active", "shelved"}
    ]
