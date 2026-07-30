"""Tests for the SQLite latest-version corpus store."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from banso.corpus import (
    CorpusDocumentStatus,
    CorpusDocumentWrite,
    DiscoveryEndpointState,
    SQLiteCorpusStore,
)


def _active_document(**overrides: object) -> CorpusDocumentWrite:
    values: dict[str, object] = {
        "source_id": "example-official",
        "url": "https://Example.org/reports/latest/?b=2&a=1#summary",
        "status": CorpusDocumentStatus.ACTIVE,
        "title": "Latest report",
        "text": "Authoritative report text.",
        "media_type": "text/html",
        "published_at": datetime(2026, 7, 28, tzinfo=timezone.utc),
        "fetched_at": datetime(2026, 7, 29, tzinfo=timezone.utc),
        "etag": '"v1"',
    }
    values.update(overrides)
    return CorpusDocumentWrite.model_validate(values)


def test_store_inserts_and_reads_latest_document(tmp_path: Path) -> None:
    with SQLiteCorpusStore(tmp_path / "corpus.db") as store:
        stored = store.upsert(_active_document())

        assert stored.canonical_url == "https://example.org/reports/latest?a=1&b=2"
        assert stored.content_hash is not None
        assert stored.created_at == stored.updated_at
        assert store.get(stored.id) == stored
        assert store.get_by_url(
            "https://example.org/reports/latest?b=2&a=1"
        ) == stored


def test_upsert_replaces_values_but_preserves_identity_and_creation_time(
    tmp_path: Path,
) -> None:
    with SQLiteCorpusStore(tmp_path / "corpus.db") as store:
        first = store.upsert(_active_document())
        second = store.upsert(
            _active_document(
                url="https://example.org/reports/latest?a=1&b=2",
                title="Revised report",
                text="Revised text.",
                etag='"v2"',
            )
        )

        assert second.id == first.id
        assert second.created_at == first.created_at
        assert second.updated_at >= first.updated_at
        assert second.title == "Revised report"
        assert second.content_hash != first.content_hash
        assert len(store.list_documents()) == 1


def test_upsert_rejects_different_source_for_existing_canonical_url(
    tmp_path: Path,
) -> None:
    with SQLiteCorpusStore(tmp_path / "corpus.db") as store:
        original = store.upsert(_active_document())

        with pytest.raises(ValueError, match="corpus document source conflict"):
            store.upsert(
                _active_document(
                    source_id="different-source",
                    url="https://example.org/reports/latest?a=1&b=2",
                    title="Conflicting ownership",
                )
            )

        assert store.get(original.id) == original
        assert len(store.list_documents()) == 1


def test_store_filters_documents_and_survives_reopen(tmp_path: Path) -> None:
    database = tmp_path / "corpus.db"
    with SQLiteCorpusStore(database) as store:
        active = store.upsert(_active_document())
        store.upsert(
            CorpusDocumentWrite(
                source_id="other-source",
                url="https://other.example.org/report",
                status=CorpusDocumentStatus.INACTIVE,
                failure_reason="robots_disallowed",
            )
        )

        assert store.list_documents(
            status=CorpusDocumentStatus.ACTIVE,
        ) == [active]

    with SQLiteCorpusStore(database) as reopened:
        assert reopened.get(active.id) == active


def test_store_rejects_invalid_url(tmp_path: Path) -> None:
    with SQLiteCorpusStore(tmp_path / "corpus.db") as store:
        with pytest.raises(ValueError, match="invalid HTTP URL"):
            store.upsert(
                CorpusDocumentWrite(source_id="source", url="/relative/report")
            )
        with pytest.raises(ValueError, match="invalid HTTP URL"):
            store.upsert(
                CorpusDocumentWrite(
                    source_id="source",
                    url="https://example.org./report",
                )
            )


def test_active_document_requires_text() -> None:
    with pytest.raises(ValueError, match="active corpus documents must contain text"):
        CorpusDocumentWrite(
            source_id="source",
            url="https://example.org/report",
            status=CorpusDocumentStatus.ACTIVE,
        )


def test_store_upserts_discovery_endpoint_validators(tmp_path: Path) -> None:
    database = tmp_path / "corpus.db"
    with SQLiteCorpusStore(database) as store:
        assert store.get_discovery_endpoint("https://example.org/feed.xml") is None
        stored = store.upsert_discovery_endpoint(
            DiscoveryEndpointState(
                url="https://Example.org/feed.xml#fragment",
                etag='"v1"',
            )
        )
        updated = store.upsert_discovery_endpoint(
            DiscoveryEndpointState(
                url="https://example.org/feed.xml",
                last_modified="Wed, 29 Jul 2026 08:00:00 GMT",
            )
        )

        assert stored.url == "https://example.org/feed.xml"
        assert updated == DiscoveryEndpointState(
            url="https://example.org/feed.xml",
            last_modified="Wed, 29 Jul 2026 08:00:00 GMT",
        )

    with SQLiteCorpusStore(database) as reopened:
        assert (
            reopened.get_discovery_endpoint("https://example.org/feed.xml")
            == updated
        )
