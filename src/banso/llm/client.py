"""LLM client interface."""

from typing import Protocol

from banso.llm.models import LLMRequest, LLMResponse


class LLMClient(Protocol):
    """Provider-independent LLM generation client."""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for an LLM request."""
        ...
