"""Simple action executor implementation for smoke testing."""

from banso.core.action import AgentAction, AgentActionType
from banso.core.observation import (
    FinishObservation,
    Observation,
    RetrievalFilterReport,
    SearchObservation,
    SearchResultMergeReport,
    SourceClassificationReport,
    StopObservation,
)
from banso.core.state import AgentState


class SimpleActionExecutor:
    """Returns deterministic observations without calling external services."""

    async def execute(self, action: AgentAction, state: AgentState) -> Observation:
        if action.type == AgentActionType.SEARCH:
            query = action.params.get("query", state.query.text)
            return SearchObservation(
                search_queries=[query],
                search_result_ids=[],
                search_result_index_updates={},
                search_result_merge_report=SearchResultMergeReport(
                    candidate_count=0,
                    new_result_count=0,
                    reused_result_count=0,
                ),
                retrieval_filter_report=RetrievalFilterReport(
                    input_count=0,
                    output_count=0,
                ),
                source_classification_report=SourceClassificationReport(
                    input_count=0,
                    recognized_count=0,
                    unknown_count=0,
                ),
            )

        if action.type == AgentActionType.FINISH:
            return FinishObservation(
                final_answer="TODO: synthesize answer",
                citations=[],
            )

        if action.type == AgentActionType.STOP:
            return StopObservation()

        raise ValueError(f"unsupported action type: {action.type.value}")
