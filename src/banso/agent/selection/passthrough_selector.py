"""Default search-result selection behavior."""

from banso.agent.selection.selector import (
    SearchResultSelection,
    SearchResultSelectionRequest,
)


class PassthroughSearchResultSelector:
    """Select every candidate while preserving provider order."""

    async def select(
        self,
        request: SearchResultSelectionRequest,
    ) -> SearchResultSelection:
        """Return all candidate IDs in their existing order."""
        return SearchResultSelection(
            selected_ids=[candidate.id for candidate in request.candidates]
        )
