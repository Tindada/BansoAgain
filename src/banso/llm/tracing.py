"""Provider-independent tracing for LLM client calls."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from banso.llm.client import LLMClient
from banso.llm.models import LLMRequest, LLMResponse
from banso.tracing.trace import start_span


class TracingLLMClient:
    """Record model inputs and outputs while delegating generation."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response and record only the model I/O boundary."""

        with start_span(
            "llm.call",
            input={"request": request},
            attributes=_trace_attributes(request.metadata),
        ) as span:
            response = await self.client.generate(request)
            span.set_output(
                {
                    "completion": response.content,
                    "provider_response": response.raw,
                    "model": response.model,
                    "usage": response.usage,
                    "response_metadata": response.metadata,
                }
            )
            return response


def _trace_attributes(metadata: Mapping[str, Any]) -> dict[str, Any]:
    trace_metadata = metadata.get("trace")
    if not isinstance(trace_metadata, Mapping):
        return {}
    return dict(trace_metadata)
