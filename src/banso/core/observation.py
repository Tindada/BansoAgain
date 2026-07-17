"""Structured observations produced by agent action execution."""

from pydantic import BaseModel, Field, JsonValue


class Observation(BaseModel):
    """Structured result returned after executing an action."""

    data: dict[str, JsonValue] = Field(default_factory=dict)
