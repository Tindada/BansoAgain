"""Tests for controlled concurrency in the news executor."""

import asyncio

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import AgentState, UserQuery
from banso.core.action import AgentAction, AgentActionType
from banso.documents import Document, EvidenceExtractionRequest, EvidenceItem
from banso.documents.fake import FakeDocumentReader
from banso.executors import NewsActionExecutor
from banso.retrieval import FakeRetrievalProvider
from banso.synthesis import FakeSynthesizer


class TrackingEvidenceExtractor:
    def __init__(self) -> None:
        self.active_count = 0
        self.max_active_count = 0

    async def extract(
        self,
        request: EvidenceExtractionRequest,
    ) -> list[EvidenceItem]:
        self.active_count += 1
        self.max_active_count = max(self.max_active_count, self.active_count)

        delay = 0.03 if request.document.title == "First" else 0.01
        await asyncio.sleep(delay)

        self.active_count -= 1
        return [
            EvidenceItem(
                document_id=request.document.id,
                claim=request.document.title,
                source_url=request.document.url,
            )
        ]


async def _run_extraction_respects_concurrency_and_document_order() -> None:
    store = InMemoryArtifactStore()
    documents = [
        Document(title="First", url="https://example.com/first", text="First"),
        Document(title="Second", url="https://example.com/second", text="Second"),
        Document(title="Third", url="https://example.com/third", text="Third"),
    ]
    state = AgentState(
        query=UserQuery(text="test query"),
        document_ids=[store.put(document) for document in documents],
    )
    extractor = TrackingEvidenceExtractor()
    executor = NewsActionExecutor(
        store=store,
        retrieval_provider=FakeRetrievalProvider(),
        document_reader=FakeDocumentReader(),
        evidence_extractor=extractor,
        synthesizer=FakeSynthesizer(),
        max_extraction_concurrency=2,
    )

    observation = await executor.execute(
        AgentAction(type=AgentActionType.EXTRACT_EVIDENCE),
        state,
    )

    evidence = [
        store.get(evidence_id, EvidenceItem)
        for evidence_id in observation.data["evidence_ids"]
    ]
    assert extractor.max_active_count == 2
    assert [item.claim for item in evidence if item is not None] == [
        "First",
        "Second",
        "Third",
    ]


def test_extraction_respects_concurrency_and_document_order() -> None:
    asyncio.run(_run_extraction_respects_concurrency_and_document_order())


def test_extraction_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        NewsActionExecutor(
            store=InMemoryArtifactStore(),
            retrieval_provider=FakeRetrievalProvider(),
            document_reader=FakeDocumentReader(),
            evidence_extractor=TrackingEvidenceExtractor(),
            synthesizer=FakeSynthesizer(),
            max_extraction_concurrency=0,
        )
