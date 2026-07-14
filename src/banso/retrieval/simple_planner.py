"""Simple search query planner implementation."""

from banso.core.state import PlannedSearch, SearchPlan
from banso.retrieval.planner import SearchPlanningRequest


class OriginalQueryPlanner:
    """Uses the original user query as a single search."""

    async def plan(self, request: SearchPlanningRequest) -> SearchPlan:
        if request.max_searches < 1:
            return SearchPlan()

        return SearchPlan(
            searches=[
                PlannedSearch(query=request.query.text, intent="general"),
            ]
        )
