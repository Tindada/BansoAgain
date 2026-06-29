"""Tests for the OpenAI SDK-backed LLM client."""

import asyncio
from types import SimpleNamespace

import pytest

from banso.llm import (
    LLMMessage,
    LLMMessageRole,
    LLMRequest,
    OpenAISDKLLMClient,
)


class FakeChatCompletions:
    def __init__(
        self,
        *,
        content: str | None = "Generated answer.",
        model: str | None = "test-model",
        usage=None,
        raw: dict | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self.content = content
        self.model = model
        self.usage = usage
        self.raw = raw or {"id": "chatcmpl-test"}

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model=self.model,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ],
            usage=self.usage,
            model_dump=lambda: self.raw,
        )


class FakeOpenAIClient:
    def __init__(self, completions: FakeChatCompletions | None = None) -> None:
        self.chat = SimpleNamespace(completions=completions or FakeChatCompletions())


async def _run_openai_sdk_client_maps_request_and_response() -> None:
    completions = FakeChatCompletions(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=14,
        )
    )
    client = OpenAISDKLLMClient(
        model="test-model",
        client=FakeOpenAIClient(completions),
    )

    response = await client.generate(
        LLMRequest(
            messages=[
                LLMMessage(
                    role=LLMMessageRole.USER,
                    content="Summarize the news.",
                )
            ],
            temperature=0.1,
            max_tokens=128,
        )
    )

    assert completions.calls == [
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Summarize the news."}],
            "temperature": 0.1,
            "max_tokens": 128,
        }
    ]
    assert response.content == "Generated answer."
    assert response.model == "test-model"
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 4
    assert response.usage.total_tokens == 14
    assert response.raw["id"] == "chatcmpl-test"


async def _run_request_model_overrides_default_model() -> None:
    completions = FakeChatCompletions()
    client = OpenAISDKLLMClient(
        model="default-model",
        client=FakeOpenAIClient(completions),
    )

    await client.generate(
        LLMRequest(
            model="request-model",
            messages=[
                LLMMessage(role=LLMMessageRole.USER, content="Use request model.")
            ],
        )
    )

    assert completions.calls[0]["model"] == "request-model"


async def _run_missing_model_raises_error() -> None:
    completions = FakeChatCompletions()
    client = OpenAISDKLLMClient(client=FakeOpenAIClient(completions))

    with pytest.raises(ValueError, match="LLM model is required"):
        await client.generate(
            LLMRequest(
                messages=[
                    LLMMessage(role=LLMMessageRole.USER, content="No model.")
                ],
            )
        )

    assert completions.calls == []


async def _run_empty_content_and_missing_usage_are_supported() -> None:
    completions = FakeChatCompletions(
        content=None,
        model=None,
        usage=None,
    )
    client = OpenAISDKLLMClient(
        model="test-model",
        client=FakeOpenAIClient(completions),
    )

    response = await client.generate(
        LLMRequest(
            messages=[
                LLMMessage(role=LLMMessageRole.USER, content="Empty content?")
            ],
        )
    )

    assert response.content == ""
    assert response.model == "test-model"
    assert response.usage is None


def test_openai_sdk_llm_client_maps_request_and_response() -> None:
    asyncio.run(_run_openai_sdk_client_maps_request_and_response())


def test_openai_sdk_llm_client_request_model_overrides_default_model() -> None:
    asyncio.run(_run_request_model_overrides_default_model())


def test_openai_sdk_llm_client_missing_model_raises_error() -> None:
    asyncio.run(_run_missing_model_raises_error())


def test_openai_sdk_llm_client_empty_content_and_missing_usage_are_supported() -> None:
    asyncio.run(_run_empty_content_and_missing_usage_are_supported())
