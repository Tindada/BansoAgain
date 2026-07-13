"""Rule-based policy implementation for smoke testing."""

from banso.core.action import AgentAction, AgentActionType
from banso.core.state import AgentState


class RuleBasedPolicy:
    """Selects a fixed search-synthesize-stop action sequence."""

    async def select_action(self, state: AgentState) -> AgentAction:
        if state.current_step == 0:
            return AgentAction(
                type=AgentActionType.SEARCH,
                params={"query": state.query.text},
                rationale="Start by searching for the user query.",
            )

        if state.current_step == 1:
            return AgentAction(
                type=AgentActionType.SYNTHESIZE,
                rationale="Synthesize after the initial search step.",
            )

        return AgentAction(
            type=AgentActionType.STOP,
            rationale="Stop after the fixed smoke-test sequence.",
        )
