"""Search-result selector interface."""

from typing import Protocol

from pydantic import BaseModel

from banso.agent.state import AgentState
from banso.retrieval.models import SearchResult


class SearchResultSelectionError(Exception):
    """A selector response could not be converted into a valid selection."""


class SearchResultSelectionRequest(BaseModel):
    """Candidate results and agent state available to a selector."""

    research_query: str
    candidates: list[SearchResult]
    state: AgentState


class SearchResultSelection(BaseModel):
    """Search results chosen for fetch and extraction."""

    selected_ids: list[str]


class SearchResultSelector(Protocol):
    """Choose which search results should enter document processing."""

    async def select(
        self,
        request: SearchResultSelectionRequest,
    ) -> SearchResultSelection:
        """Return the selected candidate result IDs."""
        ...
