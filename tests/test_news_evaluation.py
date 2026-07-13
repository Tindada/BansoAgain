"""Tests for objective news evaluation result extraction."""

import asyncio

from banso.apps.news_evaluation import (
    NewsEvaluationCase,
    extract_evaluation_result,
    load_evaluation_cases,
    summarize_evaluation_results,
)
from banso.artifacts import InMemoryArtifactStore
from banso.core import AgentRuntime, AgentState, UserQuery
from banso.documents import FakeDocumentReader, FakeEvidenceExtractor
from banso.executors import NewsActionExecutor
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import FakeRetrievalProvider, SearchRequest, SearchResult
from banso.synthesis import FakeSynthesizer


def test_load_evaluation_cases(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"case-1","category":"research","query":"Recent research?"}\n'
    )

    cases = load_evaluation_cases(path)

    assert len(cases) == 1
    assert cases[0].id == "case-1"
    assert cases[0].min_documents == 1


async def _extract_successful_evaluation_result():
    case = NewsEvaluationCase(
        id="case-1",
        category="model_release",
        query="latest AI news",
        preferred_source_types=["news"],
    )
    store = InMemoryArtifactStore()
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=NewsActionExecutor(
            store=store,
            retrieval_provider=FakeRetrievalProvider(),
            document_reader=FakeDocumentReader(),
            evidence_extractor=FakeEvidenceExtractor(),
            synthesizer=FakeSynthesizer(),
        ),
    )
    output = await runtime.run(AgentState(query=UserQuery(text=case.query)))

    return extract_evaluation_result(case, output, store)


def test_extract_evaluation_result() -> None:
    result = asyncio.run(_extract_successful_evaluation_result())

    assert result.completed is True
    assert result.passed_minimums is True
    assert result.retrieved_result_count == 1
    assert result.filtered_result_count == 1
    assert result.admitted_result_count == 1
    assert result.rejected_result_count == 0
    assert result.source_rejections == []
    assert result.document_count == 1
    assert result.evidence_count == 1
    assert result.citations == ["https://example.com/news/fake-result"]
    assert result.source_types == ["news"]
    assert result.preferred_source_type_match is True
    assert set(result.step_durations) == {
        "search",
        "read_document",
        "extract_evidence",
        "synthesize",
        "stop",
    }


class UnknownSourceRetrievalProvider:
    async def search(self, request: SearchRequest) -> list[SearchResult]:
        return [
            SearchResult(
                title="Unknown report",
                url="https://unknown.example/report",
                metadata={"score": 0.75},
            )
        ]


async def _extract_rejected_source_evaluation_result():
    case = NewsEvaluationCase(
        id="case-2",
        category="research",
        query="recent AI research",
    )
    store = InMemoryArtifactStore()
    runtime = AgentRuntime(
        policy=NewsRuleBasedPolicy(),
        executor=NewsActionExecutor(
            store=store,
            retrieval_provider=UnknownSourceRetrievalProvider(),
            document_reader=FakeDocumentReader(),
            evidence_extractor=FakeEvidenceExtractor(),
            synthesizer=FakeSynthesizer(),
        ),
    )
    output = await runtime.run(AgentState(query=UserQuery(text=case.query)))

    return extract_evaluation_result(case, output, store)


def test_extract_evaluation_result_records_source_rejections() -> None:
    result = asyncio.run(_extract_rejected_source_evaluation_result())

    assert result.retrieved_result_count == 1
    assert result.filtered_result_count == 1
    assert result.admitted_result_count == 0
    assert result.rejected_result_count == 1
    rejection = result.source_rejections[0]
    assert rejection["accepted"] is False
    assert rejection["publisher_domain"] == "unknown.example"
    assert rejection["source_type"] == "unknown"
    assert rejection["reasons"] == ["unknown_source"]


def test_summarize_evaluation_results() -> None:
    result = asyncio.run(_extract_successful_evaluation_result())
    rejected_result = asyncio.run(_extract_rejected_source_evaluation_result())

    summary = summarize_evaluation_results([result, rejected_result])

    assert summary["case_count"] == 2
    assert summary["completed_count"] == 2
    assert summary["passed_minimums_count"] == 1
    assert summary["with_documents_count"] == 1
    assert summary["with_evidence_count"] == 1
    assert summary["with_citations_count"] == 1
    assert summary["preferred_source_match_count"] == 1
    assert summary["error_count"] == 0
    assert summary["source_rejection_reasons"] == {"unknown_source": 1}
