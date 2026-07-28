"""Rule-based policy for the basic news workflow."""

from banso.core.action import AgentAction, AgentActionType
from banso.core.lifecycle import (
    active_document_count,
    eligible_extraction_document_ids,
    eligible_read_result_ids,
    remaining_document_reads,
)
from banso.core.state import AgentState


class NewsRuleBasedPolicy:
    """Selects a fixed news workflow action sequence."""

    async def select_action(self, state: AgentState) -> AgentAction:
        active_count = active_document_count(state)
        can_finish = 0 < active_count <= state.budget.max_active_documents
        if state.budget.max_steps - state.current_step <= 1:
            return (
                AgentAction(
                    type=AgentActionType.FINISH,
                    rationale="Synthesize the available research before the run ends.",
                )
                if can_finish
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
        remaining_reads = remaining_document_reads(state)
        if (
            search_index < state.budget.max_searches
            and search_index < len(state.search_plan.searches)
            and remaining_reads > 0
        ):
            search = state.search_plan.searches[search_index]
            return AgentAction(
                type=AgentActionType.SEARCH,
                params={"query": search.query, "intent": search.intent},
                rationale="Run the next planned search.",
            )

        if remaining_reads > 0 and eligible_read_result_ids(state):
            return AgentAction(
                type=AgentActionType.READ_DOCUMENT,
                rationale="Read the remaining eligible search results.",
            )

        if eligible_extraction_document_ids(state):
            return AgentAction(
                type=AgentActionType.EXTRACT_EVIDENCE,
                rationale="Extract evidence from the remaining eligible documents.",
            )

        if can_finish:
            return AgentAction(
                type=AgentActionType.FINISH,
                rationale="Synthesize the final answer and finish the workflow.",
            )

        return AgentAction(
            type=AgentActionType.STOP,
            rationale="Stop because the fixed workflow cannot make further progress.",
        )
