"""Tests for bounded news policy state views."""

from datetime import datetime, timezone

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core.state import AgentState, UserQuery
from banso.documents import Document, EvidenceItem
from banso.policies import NewsPolicyStateViewBuilder
from banso.retrieval import SearchResult, Source, SourceType


def _populated_store() -> InMemoryArtifactStore:
    store = InMemoryArtifactStore()
    source = Source(
        name="Example",
        url="https://example.com",
        type=SourceType.NEWS,
    )
    published_at = datetime(2026, 7, 17, tzinfo=timezone.utc)
    store.put(
        SearchResult(
            id="result-1",
            title="Result",
            url="https://example.com/result",
            snippet="search snippet",
            source=source,
            published_at=published_at,
            metadata={"provider_payload": "hidden"},
        )
    )
    store.put(
        Document(
            id="document-1",
            title="Document",
            url="https://example.com/document",
            text="document body",
            source=source,
            published_at=published_at,
            author="Author",
            metadata={"parser_output": "hidden"},
        )
    )
    store.put(
        EvidenceItem(
            id="evidence-1",
            document_id="document-1",
            claim="evidence claim",
            supporting_text="supporting text must stay hidden",
            source_url="https://example.com/document",
            published_at=published_at,
            confidence=0.9,
            metadata={"extractor_output": "hidden"},
        )
    )
    return store


def _state() -> AgentState:
    return AgentState(
        query=UserQuery(text="What happened?"),
        search_result_ids=["result-1"],
        document_ids=["document-1"],
        evidence_ids=["evidence-1"],
    )


def test_builds_policy_view_with_only_selected_artifact_fields() -> None:
    view = NewsPolicyStateViewBuilder(_populated_store()).build(_state())

    assert view.search_results[0].model_dump() == {
        "id": "result-1",
        "title": "Result",
        "url": "https://example.com/result",
        "snippet": "search snippet",
        "source": {
            "name": "Example",
            "url": "https://example.com",
            "type": SourceType.NEWS,
        },
        "published_at": datetime(2026, 7, 17, tzinfo=timezone.utc),
    }
    assert view.documents[0].text_preview == "document body"
    assert "text" not in view.documents[0].model_dump()
    assert view.evidence[0].claim == "evidence claim"
    assert "supporting_text" not in view.evidence[0].model_dump()


def test_preserves_state_order_and_applies_item_limits() -> None:
    store = InMemoryArtifactStore()
    for index in range(3):
        store.put(
            SearchResult(
                id=f"result-{index}",
                title=f"Result {index}",
                url=f"https://example.com/{index}",
            )
        )
    state = AgentState(
        query=UserQuery(text="query"),
        search_result_ids=["result-2", "result-0", "result-1"],
    )

    view = NewsPolicyStateViewBuilder(store, max_search_results=2).build(state)

    assert [result.id for result in view.search_results] == ["result-2", "result-0"]


def test_truncates_policy_visible_text_without_modifying_artifacts() -> None:
    store = _populated_store()
    view = NewsPolicyStateViewBuilder(
        store,
        max_snippet_chars=6,
        max_document_preview_chars=8,
        max_claim_chars=8,
    ).build(_state())

    assert view.search_results[0].snippet == "search"
    assert view.documents[0].text_preview == "document"
    assert view.evidence[0].claim == "evidence"
    stored_result = store.get("result-1", SearchResult)
    stored_document = store.get("document-1", Document)
    stored_evidence = store.get("evidence-1", EvidenceItem)
    assert stored_result is not None
    assert stored_document is not None
    assert stored_evidence is not None
    assert stored_result.snippet == "search snippet"
    assert stored_document.text == "document body"
    assert stored_evidence.claim == "evidence claim"


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (
            AgentState(
                query=UserQuery(text="query"),
                search_result_ids=["missing-result"],
            ),
            "search result artifact is missing or has the wrong type: missing-result",
        ),
        (
            AgentState(
                query=UserQuery(text="query"),
                document_ids=["result-1"],
            ),
            "document artifact is missing or has the wrong type: result-1",
        ),
    ],
)
def test_rejects_missing_or_wrongly_typed_artifact(
    state: AgentState,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        NewsPolicyStateViewBuilder(_populated_store()).build(state)


def test_validates_artifact_ids_beyond_visible_limit() -> None:
    state = AgentState(
        query=UserQuery(text="query"),
        search_result_ids=["result-1", "missing-result"],
    )

    with pytest.raises(ValueError, match="missing-result"):
        NewsPolicyStateViewBuilder(
            _populated_store(),
            max_search_results=1,
        ).build(state)


def test_view_and_input_state_are_isolated_snapshots() -> None:
    store = _populated_store()
    state = _state()
    view = NewsPolicyStateViewBuilder(store).build(state)

    state.query.text = "changed state"
    assert view.state.query.text == "What happened?"

    view.state.query.text = "changed view"
    assert state.query.text == "changed state"
    assert view.state.query.text == "changed view"

    view_source = view.search_results[0].source
    assert view_source is not None
    view_source.name = "changed source"
    stored_result = store.get("result-1", SearchResult)
    assert stored_result is not None
    assert stored_result.source is not None
    assert stored_result.source.name == "Example"


@pytest.mark.parametrize(
    "argument",
    [
        "max_search_results",
        "max_documents",
        "max_evidence",
        "max_snippet_chars",
        "max_document_preview_chars",
        "max_claim_chars",
    ],
)
def test_rejects_negative_limits(argument: str) -> None:
    with pytest.raises(ValueError, match=f"{argument} must be non-negative"):
        NewsPolicyStateViewBuilder(_populated_store(), **{argument: -1})
