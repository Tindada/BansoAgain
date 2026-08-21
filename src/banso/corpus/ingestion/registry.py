"""Loading and lookup for the curated source registry."""

import json
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from banso.source import SourceType


_DOMAIN_PATTERN = re.compile(
    r"(?=.{1,253}\Z)"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
)


class SourceRegistryError(ValueError):
    """Raised when a source registry cannot be loaded or validated."""


class TrustedSource(BaseModel):
    """A classified source and its optional ingestion scope.

    ``enabled`` controls ingestion only; every record participates in source
    classification.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1)
    source_type: SourceType
    enabled: bool = True
    allowed_domains: tuple[str, ...] = Field(min_length=1)
    allowed_path_prefixes: tuple[str, ...] = ("/",)
    feeds: tuple[str, ...] = ()
    sitemaps: tuple[str, ...] = ()

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value

    @field_validator("allowed_domains")
    @classmethod
    def _validate_domains(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        domains = tuple(value.strip().lower() for value in values)
        if any(_DOMAIN_PATTERN.fullmatch(domain) is None for domain in domains):
            raise ValueError("allowed_domains contains an invalid domain")
        if len(domains) != len(set(domains)):
            raise ValueError("allowed_domains must not contain duplicates")
        return domains

    @field_validator("allowed_path_prefixes")
    @classmethod
    def _validate_path_prefixes(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        prefixes = tuple(
            prefix if prefix == "/" else prefix.rstrip("/")
            for prefix in (value.strip() for value in values)
        )
        if any(
            not prefix.startswith("/") or "?" in prefix or "#" in prefix
            for prefix in prefixes
        ):
            raise ValueError("allowed_path_prefixes contains an invalid path")
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("allowed_path_prefixes must not contain duplicates")
        return prefixes

    @field_validator("feeds", "sitemaps")
    @classmethod
    def _validate_endpoints(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        endpoints = tuple(value.strip() for value in values)
        for endpoint in endpoints:
            parsed = _parse_http_url(endpoint)
            if parsed.fragment:
                raise ValueError("discovery endpoint must not contain a fragment")
        if len(endpoints) != len(set(endpoints)):
            raise ValueError("discovery endpoints must not contain duplicates")
        return endpoints

    @model_validator(mode="after")
    def _validate_endpoint_domains(self) -> "TrustedSource":
        if self.source_type == SourceType.UNKNOWN:
            raise ValueError("source_type must not be unknown")
        for endpoint in (*self.feeds, *self.sitemaps):
            if _parse_http_url(endpoint).hostname not in self.allowed_domains:
                raise ValueError("discovery endpoint is outside allowed_domains")
        return self

    def contains_url(self, url: str) -> bool:
        """Return whether a content URL is inside this source's approved scope."""

        try:
            parsed = _parse_http_url(url)
        except ValueError:
            return False
        if parsed.hostname not in self.allowed_domains:
            return False

        path = parsed.path or "/"
        return any(
            prefix == "/" or path == prefix or path.startswith(f"{prefix}/")
            for prefix in self.allowed_path_prefixes
        )


class SourceRegistry(BaseModel):
    """The authoritative set of sources approved for background ingestion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2]
    sources: tuple[TrustedSource, ...]

    @model_validator(mode="after")
    def _reject_registry_conflicts(self) -> "SourceRegistry":
        source_ids = [source.id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source ids must be unique")

        source_id_by_domain_key: dict[str, str] = {}
        for source in self.sources:
            for domain in source.allowed_domains:
                domain_key = _classification_domain_key(domain)
                source_id = source_id_by_domain_key.setdefault(domain_key, source.id)
                if source_id != source.id:
                    raise ValueError(
                        "classification domains must not be shared across sources"
                    )
        return self

    @classmethod
    def load(cls, path: str | Path) -> "SourceRegistry":
        """Load and validate a registry JSON file."""

        registry_path = Path(path)
        try:
            raw: Any = json.loads(registry_path.read_text(encoding="utf-8"))
            return cls.model_validate(raw)
        except OSError as error:
            raise SourceRegistryError(
                f"could not read source registry: {registry_path}"
            ) from error
        except json.JSONDecodeError as error:
            raise SourceRegistryError(
                f"source registry is not valid JSON: {registry_path}"
            ) from error
        except ValidationError as error:
            raise SourceRegistryError(
                f"source registry is invalid: {registry_path}: {error}"
            ) from error

    def get(self, source_id: str) -> TrustedSource | None:
        """Return a source by stable registry id."""

        return next((source for source in self.sources if source.id == source_id), None)

    def enabled_sources(self) -> tuple[TrustedSource, ...]:
        """Return enabled sources in registry order."""

        return tuple(source for source in self.sources if source.enabled)

    def source_type_by_domain(self) -> dict[str, SourceType]:
        """Map classification domain keys to their reviewed source types."""

        return {
            _classification_domain_key(domain): source.source_type
            for source in self.sources
            for domain in source.allowed_domains
        }

    def match_url(self, url: str) -> TrustedSource | None:
        """Return the first enabled source whose approved scope contains the URL."""

        return next(
            (
                source
                for source in self.sources
                if source.enabled and source.contains_url(url)
            ),
            None,
        )


def _parse_http_url(url: str):
    try:
        parsed = urlsplit(url)
    except ValueError as error:
        raise ValueError("expected an absolute HTTP(S) URL") from error
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.hostname.endswith(".")
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("expected an absolute HTTP(S) URL")
    return parsed


def _classification_domain_key(domain: str) -> str:
    return domain.removeprefix("www.")
