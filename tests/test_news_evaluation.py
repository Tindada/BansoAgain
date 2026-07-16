"""Tests for objective news evaluation result extraction."""

import asyncio

import pytest

from banso.apps.news_evaluation import (
    NewsEvaluationCase,
    extract_evaluation_result,
    load_evaluation_cases,
    summarize_evaluation_results,
)
from banso.artifacts import InMemoryArtifactStore
from banso.core import (
    AgentRuntime,
    AgentState,
    ExecutionBudget,
    PlannedSearch,
    SearchPlan,
    UserQuery,
)
from banso.documents import FakeDocumentReader, FakeEvidenceExtractor
from banso.executors import NewsActionExecutor
from banso.policies import NewsRuleBasedPolicy
from banso.retrieval import (
    FakeRetrievalProvider,
    SearchPlanningRequest,
    SearchRequest,
    SearchResult,
)
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
    assert result.trace_id is not None
    assert result.retrieved_result_count == 1
    assert result.filtered_result_count == 1
    assert result.classified_result_count == 1
    assert result.recognized_source_count == 1
    assert result.unknown_source_count == 0
    assert result.classification_coverage == 1.0
    assert result.document_count == 1
    assert result.evidence_count == 1
    assert result.citations == ["https://example.com/news/fake-result"]
    assert result.source_types == ["news"]
    assert result.preferred_source_type_match is True
    assert set(result.step_durations) == {
        "plan_search",
        "search",
        "read_document",
        "extract_evidence",
        "synthesize",
        "stop",
    }


class ThreeQueryPlanner:
    async def plan(self, request: SearchPlanningRequest) -> SearchPlan:
        return SearchPlan(
            searches=[
                PlannedSearch(query="AI releases", intent="official"),
                PlannedSearch(query="AI research", intent="research"),
                PlannedSearch(query="AI policy", intent="policy"),
            ][: request.max_searches]
        )


async def _extract_multi_search_evaluation_result():
    case = NewsEvaluationCase(
        id="case-multi-search",
        category="general",
        query="latest AI news",
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
            search_query_planner=ThreeQueryPlanner(),
        ),
    )
    output = await runtime.run(
        AgentState(
            query=UserQuery(text=case.query),
            budget=ExecutionBudget(max_searches=3),
        )
    )
    return extract_evaluation_result(case, output, store), output


def test_extract_evaluation_result_preserves_multiple_searches() -> None:
    result, output = asyncio.run(_extract_multi_search_evaluation_result())

    assert result.trace_id == output.trace.trace_id
    plan = output.trace.final_result.state.search_plan
    assert plan is not None
    assert [search.query for search in plan.searches] == [
        "AI releases",
        "AI research",
        "AI policy",
    ]
    search_steps = [
        step for step in output.trace.steps if step.action.type.value == "search"
    ]
    assert [step.action.params["query"] for step in search_steps] == [
        "AI releases",
        "AI research",
        "AI policy",
    ]
    assert result.retrieved_result_count == 3
    assert result.filtered_result_count == 3
    assert result.classified_result_count == 3
    assert result.recognized_source_count == 3
    assert result.unknown_source_count == 0

    search_duration = sum(
        step.duration_seconds or 0.0
        for step in output.trace.steps
        if step.action.type.value == "search"
    )
    total_duration = sum(
        step.duration_seconds or 0.0 for step in output.trace.steps
    )
    assert result.step_durations["search"] == pytest.approx(search_duration)
    assert result.total_action_seconds == pytest.approx(total_duration)


class UnknownSourceRetrievalProvider:
    async def search(self, request: SearchRequest) -> list[SearchResult]:
        return [
            SearchResult(
                title="Unknown report",
                url="https://unknown.example/report",
                metadata={"score": 0.75},
            )
        ]


async def _extract_unknown_source_evaluation_result():
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


def test_extract_evaluation_result_records_unknown_source() -> None:
    result = asyncio.run(_extract_unknown_source_evaluation_result())

    assert result.retrieved_result_count == 1
    assert result.filtered_result_count == 1
    assert result.classified_result_count == 1
    assert result.recognized_source_count == 0
    assert result.unknown_source_count == 1
    assert result.classification_coverage == 0.0
    assert result.document_count == 1
    assert len(result.source_classifications) == 1
    classification = result.source_classifications[0]
    assert classification["publisher_domain"] == "unknown.example"
    assert classification["source_type"] == "unknown"
    assert classification["classification_source"] == "unknown"


def test_summarize_evaluation_results() -> None:
    result = asyncio.run(_extract_successful_evaluation_result())
    unknown_result = asyncio.run(_extract_unknown_source_evaluation_result())

    summary = summarize_evaluation_results([result, unknown_result])

    assert summary["case_count"] == 2
    assert summary["completed_count"] == 2
    assert summary["passed_minimums_count"] == 2
    assert summary["with_documents_count"] == 2
    assert summary["with_evidence_count"] == 2
    assert summary["with_citations_count"] == 2
    assert summary["preferred_source_match_count"] == 1
    assert summary["error_count"] == 0
    assert summary["classification_coverage"] == 0.5
    assert summary["classification_source_counts"] == {
        "provider": 1,
        "unknown": 1,
    }
    assert summary["source_type_counts"] == {
        "news": 1,
        "unknown": 1,
    }
    assert summary["unknown_source_candidates"] == [
        {
            "publisher_domain": "unknown.example",
            "count": 1,
        }
    ]
