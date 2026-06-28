"""Action execution interface."""

from typing import Protocol

from banso.core.action import AgentAction
from banso.core.result import Observation
from banso.core.state import AgentState


class ActionExecutor(Protocol):
    """Executes a selected action and returns an observation."""

    async def execute(self, action: AgentAction, state: AgentState) -> Observation:
        """Execute an action against the current state."""
        ...
