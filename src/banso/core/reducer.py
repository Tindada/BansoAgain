"""State reducer interface and default implementation."""

from copy import deepcopy
from typing import Protocol

from banso.core.action import AgentAction, AgentActionType
from banso.core.lifecycle import progress_status
from banso.core.observation import Observation
from banso.core.state import (
    ActionHistoryEntry,
    AgentState,
    DocumentState,
    ExtractProgress,
    Failure,
    SearchResultState,
    SearchPlan,
)


def _update_index(
    target_index: dict[str, str],
    index_updates: object,
    label: str,
) -> None:
    if not isinstance(index_updates, dict):
        raise ValueError(f"{label} updates must be a mapping")
    if target_index.keys() & index_updates.keys():
        raise ValueError(f"{label} update contains an existing URL")
    target_index.update(index_updates)


def _apply_read_outcomes(state: AgentState, outcomes: object) -> None:
    if not isinstance(outcomes, list):
        raise ValueError("read outcomes must be a list")

    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise ValueError("read outcome must be a mapping")
        result_id = outcome["search_result_id"]
        if not isinstance(result_id, str):
            raise ValueError("read outcome search_result_id must be a string")

        result = state.search_results.get(result_id)
        if result is None:
            raise ValueError(f"read outcome contains an unknown search result: {result_id}")
        attempt_count = result.attempt_count + 1
        document_id = outcome.get("document_id")
        failure = outcome.get("failure")
        if isinstance(document_id, str) and failure is None:
            state.documents.setdefault(document_id, DocumentState())
            state.search_results[result_id] = SearchResultState(
                attempt_count=attempt_count,
                document_id=document_id,
            )
            continue
        if document_id is None and isinstance(failure, dict):
            state.search_results[result_id] = SearchResultState(
                attempt_count=attempt_count,
                failure=Failure.model_validate(failure),
            )
            continue
        raise ValueError("read outcome must contain exactly one outcome")


def _apply_extraction_outcomes(
    state: AgentState,
    outcomes: object,
    step_index: int,
) -> None:
    if not isinstance(outcomes, list):
        raise ValueError("extraction outcomes must be a list")

    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise ValueError("extraction outcome must be a mapping")
        document_id = outcome["document_id"]
        if not isinstance(document_id, str):
            raise ValueError("extraction outcome document_id must be a string")

        document = state.documents.get(document_id)
        if document is None:
            raise ValueError(f"extraction outcome contains an unknown document: {document_id}")
        previous = document.extraction
        attempt_count = previous.attempt_count + 1 if previous is not None else 1
        evidence_ids = outcome.get("evidence_ids")
        failure = outcome.get("failure")
        if isinstance(evidence_ids, list) and failure is None:
            if not all(isinstance(evidence_id, str) for evidence_id in evidence_ids):
                raise ValueError("extraction evidence_ids must contain only strings")
            if len(set(evidence_ids)) != len(evidence_ids):
                raise ValueError("extraction evidence_ids must be unique")
            document.evidence_ids = list(evidence_ids)
            document.extraction = ExtractProgress(attempt_count=attempt_count)
            if evidence_ids:
                document.lifecycle_status = "active"
            else:
                document.lifecycle_status = "unusable"
                document.lifecycle_reason = "Evidence extraction completed without evidence."
                document.lifecycle_updated_at_step = step_index
            continue
        if evidence_ids is None and isinstance(failure, dict):
            extraction_failure = Failure.model_validate(failure)
            document.extraction = ExtractProgress(
                attempt_count=attempt_count,
                failure=extraction_failure,
            )
            if (
                progress_status(
                    document.extraction,
                    state.budget.max_extraction_attempts,
                )
                == "failed"
            ):
                document.lifecycle_status = "unusable"
                document.lifecycle_reason = f"Evidence extraction failed: {extraction_failure.reason}"
                document.lifecycle_updated_at_step = step_index
            continue
        raise ValueError("extraction outcome must contain exactly one outcome")


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
                action_type=action.type,
                params=deepcopy(action.params),
                observation=observation.model_copy(deep=True),
            )
        )
        next_state.current_step += 1
        next_state.last_action = action.type

        if action.type == AgentActionType.PLAN_SEARCH:
            search_plan = observation.data.get("search_plan")
            if isinstance(search_plan, dict):
                next_state.search_plan = SearchPlan.model_validate(search_plan)

        if action.type == AgentActionType.SEARCH:
            search_result_ids = observation.data.get("search_result_ids", [])
            if not isinstance(search_result_ids, list) or not all(
                isinstance(result_id, str) for result_id in search_result_ids
            ):
                raise ValueError("search_result_ids must be a list of strings")
            for result_id in search_result_ids:
                next_state.search_results.setdefault(result_id, SearchResultState())
            _update_index(
                next_state.search_result_index,
                observation.data.get("search_result_index_updates", {}),
                "search result index",
            )

        if action.type == AgentActionType.READ_DOCUMENT:
            _apply_read_outcomes(next_state, observation.data.get("read_outcomes"))
            _update_index(
                next_state.document_index,
                observation.data.get("document_index_updates", {}),
                "document index",
            )

        if action.type == AgentActionType.EXTRACT_EVIDENCE:
            _apply_extraction_outcomes(
                next_state,
                observation.data.get("extraction_outcomes"),
                state.current_step,
            )

        if action.type == AgentActionType.CURATE_EVIDENCE:
            for status, param_name in (
                ("shelved", "shelve_document_ids"),
                ("active", "reactivate_document_ids"),
            ):
                for document_id in action.params[param_name]:
                    document = next_state.documents[document_id]
                    document.lifecycle_status = status
                    document.lifecycle_reason = action.rationale
                    document.lifecycle_updated_at_step = state.current_step

        if action.type == AgentActionType.FINISH:
            final_answer = observation.data.get("final_answer")
            if isinstance(final_answer, str):
                next_state.final_answer = final_answer
                citations = observation.data.get("citations")
                next_state.citations = (
                    [value for value in citations if isinstance(value, str)]
                    if isinstance(citations, list)
                    else []
                )

        if action.type in {AgentActionType.FINISH, AgentActionType.STOP}:
            next_state.done = True

        return next_state
