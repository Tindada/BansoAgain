"""Search query planning interface."""

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel

from banso.core.state import SearchPlan, UserQuery


class SearchPlanningRequest(BaseModel):
    """Input for producing a bounded search plan."""

    query: UserQuery
    reference_time: datetime
    max_searches: int


class SearchPlanningError(Exception):
    """Raised when a planner cannot produce a valid search plan."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class SearchQueryPlanner(Protocol):
    """Creates search queries for a user information need."""

    async def plan(self, request: SearchPlanningRequest) -> SearchPlan:
        """Return an ordered search plan."""
        ...
