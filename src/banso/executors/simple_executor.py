"""Simple action executor implementation for smoke testing."""

from banso.core.action import AgentAction, AgentActionType
from banso.core.result import Observation
from banso.core.state import AgentState


class SimpleActionExecutor:
    """Returns deterministic observations without calling external services."""

    async def execute(self, action: AgentAction, state: AgentState) -> Observation:
        if action.type == AgentActionType.SEARCH:
            query = action.params.get("query", state.query.text)
            return Observation(
                action_type=action.type,
                data={"search_queries": [query]},
            )

        if action.type == AgentActionType.SYNTHESIZE:
            return Observation(
                action_type=action.type,
                data={"final_answer": "TODO: synthesize answer"},
            )

        return Observation(action_type=action.type)
