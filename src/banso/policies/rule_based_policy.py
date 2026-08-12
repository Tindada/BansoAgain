"""Rule-based policy implementation for smoke testing."""

from banso.core.action import AgentAction, AgentActionType
from banso.core.state import AgentState


class RuleBasedPolicy:
    """Selects a fixed search-finish action sequence."""

    async def select_action(self, state: AgentState) -> AgentAction:
        if state.current_step == 0:
            return AgentAction(
                type=AgentActionType.RESEARCH,
                params={"query": state.query.text, "route": "web"},
                rationale="Run one deterministic smoke-test research action.",
            )

        if state.current_step == 1:
            return AgentAction(
                type=AgentActionType.FINISH,
                rationale="Synthesize the final answer after the initial search step.",
            )

        return AgentAction(
            type=AgentActionType.STOP,
            rationale="Stop after the fixed smoke-test sequence.",
        )
