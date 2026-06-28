"""Policy interface for selecting agent actions."""

from typing import Protocol

from banso.core.action import AgentAction
from banso.core.state import AgentState


class Policy(Protocol):
    """Selects the next action from the current agent state."""

    async def select_action(self, state: AgentState) -> AgentAction:
        """Return the next action to execute."""
        ...
