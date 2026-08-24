"""OpenAI SDK-backed LLM client implementation."""

import re
from typing import Any

from openai import APIError, AsyncOpenAI

from banso.llm.client import LLMClient
from banso.llm.errors import LLMError
from banso.llm.models import LLMRequest, LLMResponse, LLMUsage


class OpenAISDKLLMClient:
    """Calls chat completions through the official OpenAI SDK."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._client = client or AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=1,
        )

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using the SDK chat completions API."""

        model = request.model or self.model
        if not model:
            raise ValueError("LLM model is required.")

        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": message.role.value, "content": message.content}
                    for message in request.messages
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            if request.response_format is not None:
                request_kwargs["response_format"] = request.response_format
            if request.extra_body is not None:
                request_kwargs["extra_body"] = request.extra_body
            response = await self._client.chat.completions.create(**request_kwargs)
        except APIError as error:
            raise LLMError(error) from error

        content = response.choices[0].message.content or ""

        return LLMResponse(
            content=content,
            model=response.model or model,
            usage=_parse_usage(response.usage),
            raw=_to_raw_dict(response),
        )


class ThinkingModeLLMClient:
    """Apply request extras and remove thinking tags from responses."""

    _thinking_tag_pattern = re.compile(r"<think>.*?</think>", re.DOTALL)

    def __init__(
        self,
        client: LLMClient,
        *,
        request_extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.request_extra_body = request_extra_body

    async def generate(self, request: LLMRequest) -> LLMResponse:
        if self.request_extra_body is not None:
            request = request.model_copy(
                update={
                    "extra_body": {
                        **self.request_extra_body,
                        **(request.extra_body or {}),
                    }
                }
            )
        response = await self.client.generate(request)
        return response.model_copy(
            update={"content": self._strip_thinking_tags(response.content)}
        )

    def _strip_thinking_tags(self, content: str) -> str:
        return self._thinking_tag_pattern.sub("", content).strip()


def _parse_usage(usage: Any | None) -> LLMUsage | None:
    if usage is None:
        return None

    return LLMUsage(
        input_tokens=getattr(usage, "prompt_tokens", None),
        output_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def _to_raw_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return {}
