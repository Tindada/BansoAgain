"""State reducer interface and default implementation."""

from copy import deepcopy
from typing import Protocol

from banso.core.action import AgentAction, AgentActionType
from banso.core.observation import Observation
from banso.core.state import (
    ActionHistoryEntry,
    AgentState,
    ExtractProgress,
    Failure,
    ReadProgress,
    SearchPlan,
)


def _extend_unique_string_list(target: list[str], values: object) -> None:
    if isinstance(values, list):
        seen = set(target)
        for value in values:
            if isinstance(value, str) and value not in seen:
                target.append(value)
                seen.add(value)


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

        previous = state.read_progress.get(result_id)
        attempt_count = previous.attempt_count + 1 if previous is not None else 1
        document_id = outcome.get("document_id")
        failure = outcome.get("failure")
        if isinstance(document_id, str) and failure is None:
            _extend_unique_string_list(state.document_ids, [document_id])
            state.read_progress[result_id] = ReadProgress(
                attempt_count=attempt_count,
                document_id=document_id,
            )
            continue
        if document_id is None and isinstance(failure, dict):
            state.read_progress[result_id] = ReadProgress(
                attempt_count=attempt_count,
                failure=Failure.model_validate(failure),
            )
            continue
        raise ValueError("read outcome must contain exactly one outcome")


def _apply_extraction_outcomes(state: AgentState, outcomes: object) -> None:
    if not isinstance(outcomes, list):
        raise ValueError("extraction outcomes must be a list")

    for outcome in outcomes:
        if not isinstance(outcome, dict):
            raise ValueError("extraction outcome must be a mapping")
        document_id = outcome["document_id"]
        if not isinstance(document_id, str):
            raise ValueError("extraction outcome document_id must be a string")

        previous = state.extract_progress.get(document_id)
        attempt_count = previous.attempt_count + 1 if previous is not None else 1
        evidence_ids = outcome.get("evidence_ids")
        failure = outcome.get("failure")
        if isinstance(evidence_ids, list) and failure is None:
            _extend_unique_string_list(state.evidence_ids, evidence_ids)
            state.extract_progress[document_id] = ExtractProgress(
                attempt_count=attempt_count,
            )
            continue
        if evidence_ids is None and isinstance(failure, dict):
            state.extract_progress[document_id] = ExtractProgress(
                attempt_count=attempt_count,
                failure=Failure.model_validate(failure),
            )
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
            _extend_unique_string_list(
                next_state.search_result_ids,
                observation.data.get("search_result_ids"),
            )
            _update_index(
                next_state.search_result_index,
                observation.data.get("search_result_index_updates", {}),
                "search result index",
            )

        if action.type == AgentActionType.READ_DOCUMENT:
            _apply_read_outcomes(
                next_state,
                observation.data.get("read_outcomes"),
            )
            _update_index(
                next_state.document_index,
                observation.data.get("document_index_updates", {}),
                "document index",
            )

        if action.type == AgentActionType.EXTRACT_EVIDENCE:
            _apply_extraction_outcomes(
                next_state,
                observation.data.get("extraction_outcomes"),
            )

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
