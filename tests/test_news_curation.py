"""Tests for evidence-group curation and synthesis."""

import asyncio
from datetime import datetime, timezone

import pytest

from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    AgentAction,
    AgentActionType,
    AgentState,
    DefaultStateReducer,
    ExecutionBudget,
    RetrievalRoute,
    UserQuery,
)
from banso.core.state import DocumentState
from banso.documents import Document, EvidenceItem
from banso.executors import NewsActionExecutor, ResearchRouteComponents
from banso.retrieval import FakeRetrievalProvider, Source, SourceType
from banso.synthesis import Citation, SynthesisRequest, SynthesisResult


class UnusedFetcher:
    async def fetch(self, request):
        raise AssertionError("fetch should not run")


class UnusedExtractor:
    async def extract(self, request):
        raise AssertionError("extraction should not run")


class RecordingSynthesizer:
    def __init__(self) -> None:
        self.request: SynthesisRequest | None = None

    async def synthesize(self, request: SynthesisRequest) -> SynthesisResult:
        self.request = request
        return SynthesisResult(
            answer="answer",
            citations=[
                Citation(
                    reference="S1",
                    document_id="a",
                    source_url="https://example.com/a",
                )
            ],
        )


def _executor(
    store: InMemoryArtifactStore,
    synthesizer: RecordingSynthesizer | None = None,
) -> NewsActionExecutor:
    return NewsActionExecutor(
        store=store,
        research_routes={
            RetrievalRoute.WEB: ResearchRouteComponents(
                FakeRetrievalProvider(),
                UnusedFetcher(),
            )
        },
        evidence_extractor=UnusedExtractor(),
        synthesizer=synthesizer or RecordingSynthesizer(),
    )


def _state_and_store() -> tuple[AgentState, InMemoryArtifactStore]:
    store = InMemoryArtifactStore()
    documents: dict[str, DocumentState] = {}
    for identifier, status in (("a", "active"), ("b", "shelved")):
        document = Document(
            id=identifier,
            url=f"https://example.com/{identifier}",
            title=identifier.upper(),
            text=identifier,
            source=Source(name=f"Source {identifier}", type=SourceType.NEWS),
            published_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
        )
        evidence = EvidenceItem(
            id=f"evidence-{identifier}",
            document_id=identifier,
            claim=f"claim-{identifier}",
            source_url=document.url,
        )
        store.put(document)
        store.put(evidence)
        documents[identifier] = DocumentState(
            evidence_ids=[evidence.id],
            lifecycle_status=status,
        )
    return AgentState(query=UserQuery(text="query"), documents=documents), store


def test_curation_is_reversible_without_changing_artifacts() -> None:
    state, store = _state_and_store()
    executor = _executor(store)
    reducer = DefaultStateReducer()
    action = AgentAction(
        type=AgentActionType.CURATE_EVIDENCE,
        params={
            "shelve_document_ids": ["a"],
            "reactivate_document_ids": ["b"],
        },
        rationale="Prefer B.",
    )

    observation = asyncio.run(executor.execute(action, state))
    state = reducer.apply(state, action, observation)

    assert state.documents["a"].lifecycle_status == "shelved"
    assert state.documents["b"].lifecycle_status == "active"
    assert len(store.list(Document)) == 2
    assert len(store.list(EvidenceItem)) == 2


def test_curation_rejects_invalid_transition_and_active_overflow() -> None:
    state, store = _state_and_store()
    executor = _executor(store)
    invalid = AgentAction(
        type=AgentActionType.CURATE_EVIDENCE,
        params={
            "shelve_document_ids": ["b"],
            "reactivate_document_ids": [],
        },
    )
    with pytest.raises(ValueError, match="invalid lifecycle"):
        asyncio.run(executor.execute(invalid, state))

    state.budget = ExecutionBudget(max_active_documents=1)
    overflow = AgentAction(
        type=AgentActionType.CURATE_EVIDENCE,
        params={
            "shelve_document_ids": [],
            "reactivate_document_ids": ["b"],
        },
    )
    with pytest.raises(ValueError, match="active document limit"):
        asyncio.run(executor.execute(overflow, state))


def test_curation_rejects_reactivating_an_unusable_document() -> None:
    state, store = _state_and_store()
    state.documents["unusable"] = DocumentState(lifecycle_status="unusable")
    action = AgentAction(
        type=AgentActionType.CURATE_EVIDENCE,
        params={
            "shelve_document_ids": [],
            "reactivate_document_ids": ["unusable"],
        },
    )

    with pytest.raises(ValueError, match="invalid lifecycle"):
        asyncio.run(_executor(store).execute(action, state))


def test_finish_uses_only_active_documents_and_evidence() -> None:
    state, store = _state_and_store()
    synthesizer = RecordingSynthesizer()
    executor = _executor(store, synthesizer)

    observation = asyncio.run(
        executor.execute(AgentAction(type=AgentActionType.FINISH), state)
    )

    assert observation.final_answer == "answer"
    assert synthesizer.request is not None
    assert synthesizer.request.reference_time == state.reference_time
    assert len(synthesizer.request.evidence_groups) == 1
    group = synthesizer.request.evidence_groups[0]
    assert group.document_id == "a"
    assert group.title == "A"
    assert group.source_url == "https://example.com/a"
    assert group.source == Source(name="Source a", type=SourceType.NEWS)
    assert group.published_at == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert [evidence.id for evidence in group.evidence] == ["evidence-a"]


def test_finish_rejects_active_overflow() -> None:
    state, store = _state_and_store()
    state.documents["b"].lifecycle_status = "active"
    state.budget = ExecutionBudget(max_active_documents=1)

    with pytest.raises(ValueError, match="requires curation"):
        asyncio.run(
            _executor(store).execute(
                AgentAction(type=AgentActionType.FINISH),
                state,
            )
        )
