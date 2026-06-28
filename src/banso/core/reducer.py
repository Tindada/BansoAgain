"""State reducer interface and default implementation."""

from typing import Protocol

from banso.core.action import AgentAction, AgentActionType
from banso.core.result import Observation
from banso.core.state import AgentState


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

        if action.type == AgentActionType.STOP or observation.error is not None:
            next_state.done = True

        if next_state.current_step >= next_state.budget.max_steps:
            next_state.done = True

        return next_state
