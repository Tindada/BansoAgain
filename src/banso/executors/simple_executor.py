"""Simple action executor implementation for smoke testing."""

from banso.core.action import AgentAction, AgentActionType, Observation
from banso.core.state import AgentState


class SimpleActionExecutor:
    """Returns deterministic observations without calling external services."""

    async def execute(self, action: AgentAction, state: AgentState) -> Observation:
        if action.type == AgentActionType.SEARCH:
            query = action.params.get("query", state.query.text)
            return Observation(
                data={"search_queries": [query]},
            )

        if action.type == AgentActionType.FINISH:
            return Observation(
                data={"final_answer": "TODO: synthesize answer"},
            )

        return Observation()
