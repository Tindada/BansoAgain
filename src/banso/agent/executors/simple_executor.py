"""Simple action executor implementation for smoke testing."""

from banso.agent.action import AgentAction, AgentActionType, ResearchActionParams
from banso.agent.observation import (
    CompletedResearchObservation,
    FinishObservation,
    Observation,
    StopObservation,
)
from banso.agent.state import AgentState
from banso.retrieval.models import (
    RetrievalFilterReport,
    SearchResultMergeReport,
    SearchResultSelectionReport,
    SourceClassificationReport,
)


class SimpleActionExecutor:
    """Returns deterministic observations without calling external services."""

    async def execute(self, action: AgentAction, state: AgentState) -> Observation:
        if action.type == AgentActionType.RESEARCH:
            params = ResearchActionParams.model_validate(action.params)
            return CompletedResearchObservation(
                query=params.query,
                route=params.route,
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
                selection_report=SearchResultSelectionReport(
                    candidate_ids=[],
                    selected_ids=[],
                    deferred_ids=[],
                ),
                fetch_outcomes=[],
                document_index_updates={},
                extraction_outcomes=[],
            )
        if action.type == AgentActionType.FINISH:
            return FinishObservation(
                final_answer="TODO: synthesize answer",
                citations=[],
            )
        if action.type == AgentActionType.STOP:
            return StopObservation()
        raise ValueError(f"unsupported action type: {action.type.value}")
