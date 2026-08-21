"""Shared information source models."""

from enum import StrEnum

from pydantic import BaseModel


class SourceType(StrEnum):
    """High-level source category."""

    NEWS = "news"
    OFFICIAL = "official"
    RESEARCH = "research"
    LEADERBOARD = "leaderboard"
    BLOG = "blog"
    SOCIAL = "social"
    AGGREGATOR = "aggregator"
    UNKNOWN = "unknown"


class Source(BaseModel):
    """Information source metadata."""

    name: str
    url: str | None = None
    type: SourceType = SourceType.UNKNOWN
