"""Rule-based policy for the basic news workflow."""

from banso.core.action import AgentAction, AgentActionType
from banso.core.state import AgentState


class NewsRuleBasedPolicy:
    """Selects a fixed news workflow action sequence."""

    async def select_action(self, state: AgentState) -> AgentAction:
        if state.search_plan is None:
            return AgentAction(
                type=AgentActionType.PLAN_SEARCH,
                rationale="Plan searches for the user query.",
            )

        search_index = len(state.search_queries)
        if (
            search_index < state.budget.max_searches
            and search_index < len(state.search_plan.searches)
        ):
            search = state.search_plan.searches[search_index]
            return AgentAction(
                type=AgentActionType.SEARCH,
                params={"query": search.query, "intent": search.intent},
                rationale="Run the next planned search.",
            )

        if state.last_action in {
            None,
            AgentActionType.PLAN_SEARCH,
            AgentActionType.SEARCH,
        }:
            return AgentAction(
                type=AgentActionType.READ_DOCUMENT,
                rationale="Read documents from the collected search results.",
            )

        if state.last_action == AgentActionType.READ_DOCUMENT:
            return AgentAction(
                type=AgentActionType.EXTRACT_EVIDENCE,
                rationale="Extract evidence from the collected documents.",
            )

        if state.last_action == AgentActionType.EXTRACT_EVIDENCE:
            return AgentAction(
                type=AgentActionType.SYNTHESIZE,
                rationale="Synthesize an answer from the collected evidence.",
            )

        return AgentAction(
            type=AgentActionType.STOP,
            rationale="Stop after completing the fixed news workflow.",
        )
