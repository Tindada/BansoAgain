"""Rule-based policy for the basic news workflow."""

from banso.core.action import AgentAction, AgentActionType
from banso.core.lifecycle import (
    eligible_extraction_document_ids,
    eligible_read_result_ids,
)
from banso.core.state import AgentState


class NewsRuleBasedPolicy:
    """Selects a fixed news workflow action sequence."""

    async def select_action(self, state: AgentState) -> AgentAction:
        has_sources = bool(state.document_ids or state.evidence_ids)
        if state.budget.max_steps - state.current_step <= 1:
            return (
                AgentAction(
                    type=AgentActionType.FINISH,
                    rationale="Synthesize the available research before the run ends.",
                )
                if has_sources
                else AgentAction(
                    type=AgentActionType.STOP,
                    rationale="Stop because no step remains to collect usable sources.",
                )
            )

        if state.search_plan is None:
            return AgentAction(
                type=AgentActionType.PLAN_SEARCH,
                rationale="Plan searches for the user query.",
            )

        search_index = sum(
            entry.action_type == AgentActionType.SEARCH
            for entry in state.action_history
        )
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

        if eligible_read_result_ids(state):
            return AgentAction(
                type=AgentActionType.READ_DOCUMENT,
                rationale="Read the remaining eligible search results.",
            )

        if eligible_extraction_document_ids(state):
            return AgentAction(
                type=AgentActionType.EXTRACT_EVIDENCE,
                rationale="Extract evidence from the remaining eligible documents.",
            )

        if has_sources:
            return AgentAction(
                type=AgentActionType.FINISH,
                rationale="Synthesize the final answer and finish the workflow.",
            )

        return AgentAction(
            type=AgentActionType.STOP,
            rationale="Stop because the fixed workflow cannot make further progress.",
        )
