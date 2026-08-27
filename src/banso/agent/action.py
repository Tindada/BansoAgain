"""Agent action models."""

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_DOMAIN_PATTERN = re.compile(
    r"(?=^.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
)


class RetrievalRoute(StrEnum):
    """Semantic retrieval routes available to a research action."""

    WEB = "web"
    LOCAL = "local"


class AgentActionType(StrEnum):
    """Supported action types for the agent loop."""

    RESEARCH = "research"
    REWRITE_SCRATCH = "rewrite_scratch"
    FINISH = "finish"
    STOP = "stop"


class ResearchActionParams(BaseModel):
    """Strict parameters accepted by a research action."""

    model_config = ConfigDict(extra="forbid")

    query: str
    route: RetrievalRoute
    source_domains: list[str] | None = None

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("query must be non-empty")
        return query

    @field_validator("source_domains")
    @classmethod
    def normalize_source_domains(
        cls,
        value: list[str] | None,
    ) -> list[str] | None:
        if not value:
            return None
        domains = [domain.strip().lower() for domain in value]
        if any(not _DOMAIN_PATTERN.fullmatch(domain) for domain in domains):
            raise ValueError("source_domains must contain bare domain names")
        if len(set(domains)) != len(domains):
            raise ValueError("source_domains must contain unique domains")
        return domains

    @model_validator(mode="after")
    def validate_source_domains_route(self) -> "ResearchActionParams":
        if self.source_domains is not None and self.route != RetrievalRoute.WEB:
            raise ValueError("source_domains is only supported by the web route")
        return self


class AgentAction(BaseModel):
    """A structured decision selected by a policy."""

    type: AgentActionType
    params: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None
