"""Tests for in-memory artifact snapshot semantics."""

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.documents import Document
from banso.retrieval import SearchResult


def test_store_rejects_duplicate_artifact_id_without_overwriting() -> None:
    store = InMemoryArtifactStore()
    original = SearchResult(
        id="result-1",
        title="Original",
        url="https://example.com/original",
    )
    replacement = SearchResult(
        id="result-1",
        title="Replacement",
        url="https://example.com/replacement",
    )
    store.put(original)

    with pytest.raises(ValueError, match="artifact already exists: result-1"):
        store.put(replacement)

    assert store.get("result-1", SearchResult) == original


def test_store_snapshots_artifact_on_put_and_get() -> None:
    store = InMemoryArtifactStore()
    result = SearchResult(
        id="result-1",
        title="Original",
        url="https://example.com/original",
        metadata={"labels": ["initial"]},
    )
    store.put(result)

    result.title = "Changed after put"
    result.metadata["labels"].append("changed-after-put")
    first_snapshot = store.get("result-1", SearchResult)
    assert first_snapshot is not None
    assert first_snapshot.title == "Original"
    assert first_snapshot.metadata == {"labels": ["initial"]}

    first_snapshot.title = "Changed after get"
    first_snapshot.metadata["labels"].append("changed-after-get")
    second_snapshot = store.get("result-1", SearchResult)
    assert second_snapshot is not None
    assert second_snapshot.title == "Original"
    assert second_snapshot.metadata == {"labels": ["initial"]}


def test_store_list_preserves_order_and_returns_snapshots() -> None:
    store = InMemoryArtifactStore()
    first = SearchResult(
        id="result-1",
        title="First",
        url="https://example.com/first",
    )
    second = SearchResult(
        id="result-2",
        title="Second",
        url="https://example.com/second",
    )
    store.put(first)
    store.put(
        Document(
            id="document-1",
            title="Document",
            url="https://example.com/document",
            text="Text",
        )
    )
    store.put(second)

    snapshots = store.list(SearchResult)
    assert [result.id for result in snapshots] == ["result-1", "result-2"]

    snapshots[0].title = "Changed after list"
    stored_first = store.get("result-1", SearchResult)
    assert stored_first is not None
    assert stored_first.title == "First"


def test_store_returns_none_for_missing_or_wrong_artifact_type() -> None:
    store = InMemoryArtifactStore()
    store.put(
        SearchResult(
            id="result-1",
            title="Result",
            url="https://example.com/result",
        )
    )

    assert store.get("missing", SearchResult) is None
    assert store.get("result-1", Document) is None
