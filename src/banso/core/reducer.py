"""State reducer interface and default implementation."""

from typing import Protocol

from banso.core.action import AgentAction, AgentActionType
from banso.core.result import Observation
from banso.core.state import AgentState, SearchPlan


def _extend_string_list(target: list[str], values: object) -> None:
    if isinstance(values, list):
        target.extend(value for value in values if isinstance(value, str))


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
        next_state.current_step += 1
        next_state.last_action = action.type

        if action.type == AgentActionType.PLAN_SEARCH:
            search_plan = observation.data.get("search_plan")
            if isinstance(search_plan, dict):
                next_state.search_plan = SearchPlan.model_validate(search_plan)

        _extend_string_list(
            next_state.search_queries,
            observation.data.get("search_queries"),
        )
        _extend_string_list(
            next_state.search_result_ids,
            observation.data.get("search_result_ids"),
        )
        _extend_string_list(
            next_state.document_ids,
            observation.data.get("document_ids"),
        )
        _extend_string_list(
            next_state.evidence_ids,
            observation.data.get("evidence_ids"),
        )

        if action.type == AgentActionType.SYNTHESIZE:
            final_answer = observation.data.get("final_answer")
            if isinstance(final_answer, str):
                next_state.final_answer = final_answer

        if action.type == AgentActionType.STOP:
            next_state.done = True

        return next_state
