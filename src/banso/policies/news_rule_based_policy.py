"""Rule-based policy for the basic news workflow."""

from banso.core.action import AgentAction, AgentActionType
from banso.core.state import AgentState


class NewsRuleBasedPolicy:
    """Selects a fixed news workflow action sequence."""

    async def select_action(self, state: AgentState) -> AgentAction:
        if state.current_step == 0:
            return AgentAction(
                type=AgentActionType.SEARCH,
                params={"query": state.query.text},
                rationale="Search for the original user query.",
            )

        if state.current_step == 1:
            return AgentAction(
                type=AgentActionType.READ_DOCUMENT,
                rationale="Read documents from the collected search results.",
            )

        if state.current_step == 2:
            return AgentAction(
                type=AgentActionType.EXTRACT_EVIDENCE,
                rationale="Extract evidence from the collected documents.",
            )

        if state.current_step == 3:
            return AgentAction(
                type=AgentActionType.SYNTHESIZE,
                rationale="Synthesize an answer from the collected evidence.",
            )

        return AgentAction(
            type=AgentActionType.STOP,
            rationale="Stop after completing the fixed news workflow.",
        )
