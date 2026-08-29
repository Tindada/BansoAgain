"""LLM client interface."""

from collections.abc import Callable
from typing import Protocol

from banso.llm.models import LLMRequest, LLMResponse


class LLMClient(Protocol):
    """Provider-independent LLM generation client."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for an LLM request."""
        ...


async def generate_validated[T, E: Exception](
    client: LLMClient,
    request: LLMRequest,
    validate: Callable[[str], T],
    *,
    error_type: type[E],
) -> tuple[LLMResponse, T]:
    """Generate and validate a response, retrying one validation failure."""
    response = await client.generate(request)
    try:
        return response, validate(response.content)
    except error_type:
        response = await client.generate(request)
        return response, validate(response.content)
