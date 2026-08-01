"""Shared source classification types."""

from enum import StrEnum


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
