"""Search query planning interface."""

from typing import Protocol

from pydantic import BaseModel

from banso.core.state import SearchPlan, UserQuery


class SearchPlanningRequest(BaseModel):
    """Input for producing a bounded search plan."""

    query: UserQuery
    max_searches: int


class SearchQueryPlanner(Protocol):
    """Creates search queries for a user information need."""

    async def plan(self, request: SearchPlanningRequest) -> SearchPlan:
        """Return an ordered search plan."""
        ...
