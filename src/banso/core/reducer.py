"""State reducer interface and default implementation."""

from typing import Protocol

from banso.core.action import AgentAction, AgentActionType
from banso.core.observation import (
    CompletedResearchObservation,
    CurateEvidenceObservation,
    ExtractionFailure,
    ExtractionOutcome,
    ExtractionSuccess,
    FetchFailure,
    FetchOutcome,
    FetchSuccess,
    FinishObservation,
    Observation,
)
from banso.core.state import (
    ActionHistoryEntry,
    AgentState,
    DocumentState,
    Failure,
    SearchResultState,
)


def _update_index(
    target_index: dict[str, str],
    index_updates: dict[str, str],
    label: str,
) -> None:
    if target_index.keys() & index_updates.keys():
        raise ValueError(f"{label} update contains an existing URL")
    target_index.update(index_updates)


def _apply_fetch_outcomes(
    state: AgentState,
    outcomes: list[FetchOutcome],
) -> None:
    for outcome in outcomes:
        result_id = outcome.search_result_id
        result = state.search_results.get(result_id)
        if result is None:
            raise ValueError(f"fetch outcome contains an unknown search result: {result_id}")
        if isinstance(outcome, FetchSuccess):
            state.documents.setdefault(outcome.document_id, DocumentState())
            state.search_results[result_id] = SearchResultState(
                document_id=outcome.document_id,
            )
            continue
        if isinstance(outcome, FetchFailure):
            state.search_results[result_id] = SearchResultState(
                failure=Failure(
                    reason=outcome.failure.reason,
                    status_code=outcome.failure.status_code,
                ),
            )
            continue
        raise AssertionError(f"unexpected fetch outcome: {type(outcome).__name__}")


def _apply_extraction_outcomes(
    state: AgentState,
    outcomes: list[ExtractionOutcome],
    step_index: int,
) -> None:
    for outcome in outcomes:
        document_id = outcome.document_id
        document = state.documents.get(document_id)
        if document is None:
            raise ValueError(f"extraction outcome contains an unknown document: {document_id}")
        if isinstance(outcome, ExtractionSuccess):
            document.evidence_ids = list(outcome.evidence_ids)
            if outcome.evidence_ids:
                document.lifecycle_status = "active"
            else:
                document.lifecycle_status = "unusable"
                document.lifecycle_reason = "Evidence extraction completed without evidence."
                document.lifecycle_updated_at_step = step_index
            continue
        if isinstance(outcome, ExtractionFailure):
            document.lifecycle_status = "unusable"
            document.lifecycle_reason = (
                f"Evidence extraction failed: {outcome.failure.reason}"
            )
            document.lifecycle_updated_at_step = step_index
            continue
        raise AssertionError(f"unexpected extraction outcome: {type(outcome).__name__}")


class StateReducer(Protocol):
    """Applies action observations to produce the next state."""

    def apply(
        self,
        state: AgentState,
        action: AgentAction,
        observation: Observation,
    ) -> AgentState:
        """Return the next state after an action observation."""
        ...


class DefaultStateReducer:
    """Minimal reducer for step counting and stop conditions."""

    def apply(
        self,
        state: AgentState,
        action: AgentAction,
        observation: Observation,
    ) -> AgentState:
        next_state = state.model_copy(deep=True)
        next_state.action_history.append(
            ActionHistoryEntry(
                step_index=state.current_step,
                action=action.model_copy(deep=True),
                observation=observation.model_copy(deep=True),
            )
        )
        next_state.current_step += 1
        next_state.last_action = action.type

        if isinstance(observation, CompletedResearchObservation):
            for result_id in observation.search_result_ids:
                next_state.search_results.setdefault(result_id, SearchResultState())
            _update_index(
                next_state.search_result_index,
                observation.search_result_index_updates,
                "search result index",
            )
            _apply_fetch_outcomes(next_state, observation.fetch_outcomes)
            _update_index(
                next_state.document_index,
                observation.document_index_updates,
                "document index",
            )
            _apply_extraction_outcomes(
                next_state,
                observation.extraction_outcomes,
                state.current_step,
            )
        elif isinstance(observation, CurateEvidenceObservation):
            for status, param_name in (
                ("shelved", "shelve_document_ids"),
                ("active", "reactivate_document_ids"),
            ):
                for document_id in action.params[param_name]:
                    document = next_state.documents[document_id]
                    document.lifecycle_status = status
                    document.lifecycle_reason = action.rationale
                    document.lifecycle_updated_at_step = state.current_step
        elif isinstance(observation, FinishObservation):
            next_state.final_answer = observation.final_answer
            next_state.citations = list(observation.citations)

        if action.type in {AgentActionType.FINISH, AgentActionType.STOP}:
            next_state.done = True

        return next_state
