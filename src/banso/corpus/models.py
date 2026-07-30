"""Models for the latest-version local document corpus."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CorpusDocumentStatus(StrEnum):
    """Background-ingestion state, independent of Agent document lifecycle."""

    DISCOVERED = "discovered"
    ACTIVE = "active"
    INACTIVE = "inactive"


class CorpusDocumentWrite(BaseModel):
    """Latest document values supplied to a corpus store."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    url: str = Field(min_length=1)
    status: CorpusDocumentStatus = CorpusDocumentStatus.DISCOVERED
    title: str | None = None
    text: str | None = None
    media_type: str | None = None
    published_at: datetime | None = None
    fetched_at: datetime | None = None
    etag: str | None = None
    last_modified: str | None = None
    failure_reason: str | None = None

    @field_validator("url")
    @classmethod
    def _strip_url(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("url must not be blank")
        return value

    @field_validator(
        "title",
        "media_type",
        "etag",
        "last_modified",
        "failure_reason",
    )
    @classmethod
    def _strip_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("published_at", "fetched_at")
    @classmethod
    def _require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("corpus timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def _active_documents_require_text(self) -> "CorpusDocumentWrite":
        if self.status == CorpusDocumentStatus.ACTIVE and (
            self.text is None or not self.text.strip()
        ):
            raise ValueError("active corpus documents must contain text")
        return self


class CorpusDocument(CorpusDocumentWrite):
    """A persisted latest-version corpus document."""

    id: str
    canonical_url: str
    content_hash: str | None
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def _require_stored_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("corpus timestamps must include a timezone")
        return value
