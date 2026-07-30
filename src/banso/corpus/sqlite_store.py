"""SQLite implementation of the latest-version corpus store."""

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from urllib.parse import urlsplit
from uuid import uuid4

from banso.corpus.models import (
    CorpusDocument,
    CorpusDocumentStatus,
    CorpusDocumentWrite,
    DiscoveryEndpointState,
)
from banso.retrieval.url_utils import normalize_url

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS corpus_documents (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    url TEXT NOT NULL,
    canonical_url TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('discovered', 'active', 'inactive')),
    title TEXT,
    text TEXT,
    media_type TEXT,
    published_at TEXT,
    fetched_at TEXT,
    etag TEXT,
    last_modified TEXT,
    content_hash TEXT,
    failure_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS corpus_documents_status_idx
    ON corpus_documents (status);

CREATE TABLE IF NOT EXISTS discovery_endpoints (
    url TEXT PRIMARY KEY,
    etag TEXT,
    last_modified TEXT
);
"""


class SQLiteCorpusStore:
    """SQLite-backed authoritative store for the latest document body."""

    def __init__(self, path: str | Path) -> None:
        self._connection = sqlite3.connect(str(path))
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA_SQL)

    def __enter__(self) -> "SQLiteCorpusStore":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def upsert(self, document: CorpusDocumentWrite) -> CorpusDocument:
        canonical_url = _canonical_http_url(document.url)
        now = datetime.now(timezone.utc).isoformat()
        values = document.model_dump(mode="json")
        values.update(
            id=str(uuid4()),
            canonical_url=canonical_url,
            content_hash=_content_hash(document.text),
            created_at=now,
            updated_at=now,
        )
        with self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO corpus_documents (
                    id, source_id, url, canonical_url, status, title, text,
                    media_type, published_at, fetched_at, etag, last_modified,
                    content_hash, failure_reason, created_at, updated_at
                ) VALUES (
                    :id, :source_id, :url, :canonical_url, :status, :title, :text,
                    :media_type, :published_at, :fetched_at, :etag, :last_modified,
                    :content_hash, :failure_reason, :created_at, :updated_at
                )
                ON CONFLICT(canonical_url) DO UPDATE SET
                    url = excluded.url,
                    status = excluded.status,
                    title = excluded.title,
                    text = excluded.text,
                    media_type = excluded.media_type,
                    published_at = excluded.published_at,
                    fetched_at = excluded.fetched_at,
                    etag = excluded.etag,
                    last_modified = excluded.last_modified,
                    content_hash = excluded.content_hash,
                    failure_reason = excluded.failure_reason,
                    updated_at = excluded.updated_at
                WHERE corpus_documents.source_id = excluded.source_id
                RETURNING *
                """,
                values,
            )
            stored_row = cursor.fetchone()
            if stored_row is None:
                existing_row = self._connection.execute(
                    """
                    SELECT source_id
                    FROM corpus_documents
                    WHERE canonical_url = ?
                    """,
                    (canonical_url,),
                ).fetchone()
                if existing_row is None:
                    raise RuntimeError(
                        f"failed to store corpus document: {canonical_url}"
                    )
                raise ValueError(
                    f"corpus document source conflict for {canonical_url}: "
                    f"existing source is {existing_row['source_id']!r}, "
                    f"incoming source is {document.source_id!r}"
                )
        return _row_to_document(stored_row)

    def get(self, document_id: str) -> CorpusDocument | None:
        row = self._connection.execute(
            "SELECT * FROM corpus_documents WHERE id = ?",
            (document_id,),
        ).fetchone()
        return _row_to_document(row) if row is not None else None

    def get_by_url(self, url: str) -> CorpusDocument | None:
        canonical_url = _canonical_http_url(url)
        row = self._connection.execute(
            "SELECT * FROM corpus_documents WHERE canonical_url = ?",
            (canonical_url,),
        ).fetchone()
        return _row_to_document(row) if row is not None else None

    def list_documents(
        self,
        *,
        status: CorpusDocumentStatus | None = None,
    ) -> list[CorpusDocument]:
        values: list[str] = []
        query = "SELECT * FROM corpus_documents"
        if status is not None:
            query += " WHERE status = ?"
            values.append(status.value)
        query += " ORDER BY created_at, id"

        rows = self._connection.execute(query, values).fetchall()
        return [_row_to_document(row) for row in rows]

    def get_discovery_endpoint(
        self,
        url: str,
    ) -> DiscoveryEndpointState | None:
        canonical_url = _canonical_http_url(url)
        row = self._connection.execute(
            "SELECT * FROM discovery_endpoints WHERE url = ?",
            (canonical_url,),
        ).fetchone()
        return (
            DiscoveryEndpointState.model_validate(dict(row))
            if row is not None
            else None
        )

    def upsert_discovery_endpoint(
        self,
        state: DiscoveryEndpointState,
    ) -> DiscoveryEndpointState:
        values = state.model_dump()
        values["url"] = _canonical_http_url(state.url)
        with self._connection:
            row = self._connection.execute(
                """
                INSERT INTO discovery_endpoints (url, etag, last_modified)
                VALUES (:url, :etag, :last_modified)
                ON CONFLICT(url) DO UPDATE SET
                    etag = excluded.etag,
                    last_modified = excluded.last_modified
                RETURNING *
                """,
                values,
            ).fetchone()
        if row is None:
            raise RuntimeError(f"failed to store discovery endpoint: {state.url}")
        return DiscoveryEndpointState.model_validate(dict(row))


def _canonical_http_url(url: str) -> str:
    try:
        parsed = urlsplit(url.strip())
    except ValueError as error:
        raise ValueError(f"invalid HTTP URL: {url!r}") from error
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.hostname.endswith(".")
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(f"invalid HTTP URL: {url!r}")
    return normalize_url(url)


def _content_hash(text: str | None) -> str | None:
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row_to_document(row: sqlite3.Row) -> CorpusDocument:
    return CorpusDocument.model_validate(dict(row))
