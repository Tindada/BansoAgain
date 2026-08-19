"""LLM request and response models."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class LLMMessageRole(StrEnum):
    """Supported LLM message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LLMMessage(BaseModel):
    """A single chat-style LLM message."""

    role: LLMMessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMRequest(BaseModel):
    """Provider-independent LLM generation request."""

    messages: list[LLMMessage]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMUsage(BaseModel):
    """Token usage reported by an LLM provider."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class LLMResponse(BaseModel):
    """Provider-independent LLM generation response."""

    content: str
    model: str | None = None
    usage: LLMUsage | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
