"""Fake LLM client for tests and local smoke runs."""

from banso.llm.models import LLMRequest, LLMResponse, LLMUsage


class FakeLLMClient:
    """Returns deterministic responses without calling an external model."""

    def __init__(self, content: str = "Fake LLM response.") -> None:
        self.content = content
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        input_tokens = sum(
            len(message.content.split()) for message in request.messages
        )
        output_tokens = len(self.content.split())

        return LLMResponse(
            content=self.content,
            model=request.model or "fake-llm",
            usage=LLMUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
