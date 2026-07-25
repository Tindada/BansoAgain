"""Shared resource lifecycle decisions."""

from typing import Literal

from banso.core.state import AgentState, ExtractProgress, ReadProgress

LifecycleStatus = Literal["pending", "succeeded", "retryable", "failed"]


def read_status(state: AgentState, result_id: str) -> LifecycleStatus:
    """Return the current read status for one search result."""
    progress = state.read_progress.get(result_id)
    if progress is None:
        return "pending"
    return _progress_status(progress, state.budget.max_read_attempts)


def extraction_status(state: AgentState, document_id: str) -> LifecycleStatus:
    """Return the current extraction status for one document."""
    progress = state.extract_progress.get(document_id)
    if progress is None:
        return "pending"
    return _progress_status(progress, state.budget.max_extraction_attempts)


def remaining_document_count(state: AgentState) -> int:
    """Return the number of unique documents the run may still collect."""
    return max(state.budget.max_documents_to_read - len(state.document_ids), 0)


def eligible_read_result_ids(state: AgentState) -> list[str]:
    """Return unread results followed by retryable read failures."""
    if remaining_document_count(state) == 0:
        return []
    pending: list[str] = []
    retryable: list[str] = []
    for result_id in state.search_result_ids:
        status = read_status(state, result_id)
        if status == "pending":
            pending.append(result_id)
        elif status == "retryable":
            retryable.append(result_id)
    return [*pending, *retryable]


def eligible_extraction_document_ids(state: AgentState) -> list[str]:
    """Return unprocessed documents followed by retryable extraction failures."""
    pending: list[str] = []
    retryable: list[str] = []
    for document_id in state.document_ids:
        status = extraction_status(state, document_id)
        if status == "pending":
            pending.append(document_id)
        elif status == "retryable":
            retryable.append(document_id)
    return [*pending, *retryable]


def _progress_status(
    progress: ReadProgress | ExtractProgress,
    max_attempts: int,
) -> LifecycleStatus:
    if progress.failure is None:
        return "succeeded"
    if progress.failure.retryable and progress.attempt_count < max_attempts:
        return "retryable"
    return "failed"
