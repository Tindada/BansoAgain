"""Action execution interface."""

from typing import Protocol

from banso.agent.action import AgentAction
from banso.agent.observation import Observation
from banso.agent.state import AgentState


class ActionExecutor(Protocol):
    """Executes a selected action and returns an observation."""

    async def execute(self, action: AgentAction, state: AgentState) -> Observation:
        """Execute an action against the current state."""
        ...
