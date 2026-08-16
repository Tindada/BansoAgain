"""Retrieval provider interface."""

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from banso.http_errors import is_retryable_http_status
from banso.retrieval.models import SearchResult


RetrievalFailureReason = Literal[
    "http_status",
    "transport",
    "invalid_response",
]


class RetrievalError(Exception):
    """A known external failure produced by a retrieval provider."""

    def __init__(
        self,
        *,
        provider: str,
        reason: RetrievalFailureReason,
        message: str,
        source_error_type: str,
        status_code: int | None = None,
    ) -> None:
        self.provider = provider
        self.reason = reason
        self.message = message
        self.source_error_type = source_error_type
        self.status_code = status_code
        super().__init__(message)

    @property
    def retryable(self) -> bool:
        """Return whether repeating the same retrieval may recover."""
        if self.reason == "transport":
            return True
        return (
            self.reason == "http_status"
            and self.status_code is not None
            and is_retryable_http_status(self.status_code)
        )


class SearchRequest(BaseModel):
    """Structured search request passed to retrieval providers."""

    query: str
    max_results: int = 10
    language: str | None = None
    region: str | None = None
    time_range: str | None = None
    source_domains: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalProvider(Protocol):
    """Search provider abstraction."""

    async def search(self, request: SearchRequest) -> list[SearchResult]:
        """Return search results for a request."""
        ...
