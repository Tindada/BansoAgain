"""Tests for news runtime evaluation extraction."""

import asyncio
import json

from banso.apps.news_evaluation import (
    NewsEvaluationCase,
    NewsEvaluationResult,
    extract_evaluation_result,
    load_evaluation_cases,
    summarize_evaluation_results,
)
from banso.artifacts.store import InMemoryArtifactStore
from banso.agent.action import AgentAction, AgentActionType
from banso.agent.observation import (
    CompletedResearchObservation,
    FailedResearchObservation,
    FinishObservation,
    StopObservation,
)
from banso.agent.runtime import AgentRuntime
from banso.agent.state import AgentState, UserQuery
from banso.documents.models import Document, DocumentEvidence
from banso.retrieval.models import SearchResult
from banso.retrieval.models import (
    RetrievalFilterReport,
    SearchResultMergeReport,
    SearchResultSelectionReport,
    SourceClassificationReport,
)
from banso.source import Source, SourceType
from banso.synthesis.synthesizer import Citation
from banso.tracing.trace import InMemoryTraceSink, Tracer


def test_load_evaluation_cases(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"id": "case", "category": "news", "query": "question"})
        + "\n"
    )

    cases = load_evaluation_cases(path)

    assert cases == [
        NewsEvaluationCase(id="case", category="news", query="question")
    ]


class Policy:
    def __init__(
        self,
        terminal_action: AgentActionType = AgentActionType.FINISH,
    ) -> None:
        self.terminal_action = terminal_action

    async def select_action(self, state: AgentState) -> AgentAction:
        if state.current_step == 0:
            return AgentAction(
                type=AgentActionType.RESEARCH,
                params={"query": "focused", "route": "web"},
            )
        return AgentAction(type=self.terminal_action)


class Executor:
    def __init__(
        self,
        research_failure: FailedResearchObservation | None = None,
    ) -> None:
        self.research_failure = research_failure

    async def execute(self, action: AgentAction, state: AgentState):
        if action.type == AgentActionType.FINISH:
            return FinishObservation(
                final_answer="answer",
                citations=[
                    Citation(
                        reference="S1",
                        document_id="document",
                        source_url="https://example.com/article",
                    )
                ],
            )
        if action.type == AgentActionType.STOP:
            return StopObservation()
        if self.research_failure is not None:
            return self.research_failure
        result_ids = ["result"]
        return CompletedResearchObservation(
            query="focused",
            route="web",
            search_result_ids=result_ids,
            search_result_index_updates={"https://example.com/article": "result"},
            search_result_merge_report=SearchResultMergeReport(
                candidate_count=len(result_ids),
                new_result_count=len(result_ids),
                reused_result_count=0,
            ),
            retrieval_filter_report=RetrievalFilterReport(
                input_count=2,
                output_count=len(result_ids),
                dropped_invalid_url=1,
            ),
            source_classification_report=SourceClassificationReport(
                input_count=len(result_ids),
                recognized_count=len(result_ids),
                unknown_count=0,
            ),
            selection_report=SearchResultSelectionReport(
                candidate_ids=result_ids,
                selected_ids=result_ids,
            ),
            fetch_outcomes=[
                {
                    "status": "success",
                    "search_result_id": "result",
                    "document_id": "document",
                }
            ],
            document_index_updates={"https://example.com/article": "document"},
            extraction_outcomes=[
                {
                    "status": "success",
                    "document_id": "document",
                    "evidence_id": "evidence",
                }
            ],
        )


def test_extract_evaluation_result_reads_composite_research_observation() -> None:
    store = InMemoryArtifactStore()
    source = Source(name="Example", type=SourceType.NEWS)
    store.put(
        SearchResult(
            id="result",
            title="Result",
            url="https://example.com/article",
            source=source,
        )
    )
    store.put(
        Document(
            id="document",
            title="Document",
            url="https://example.com/article",
            text="body",
            source=source,
        )
    )
    store.put(
        DocumentEvidence(
            id="evidence",
            document_id="document",
            text="claim",
        )
    )
    sink = InMemoryTraceSink()
    runtime = AgentRuntime(Policy(), Executor(), tracer=Tracer(sink))
    output = asyncio.run(runtime.run(AgentState(query=UserQuery(text="question"))))
    case = NewsEvaluationCase(
        id="case",
        category="news",
        query="question",
        preferred_source_types=["news"],
    )

    result = extract_evaluation_result(
        case,
        output,
        store,
        sink.get_trace(output.trace_id),
    )

    assert result.passed_minimums is True
    assert result.retrieved_result_count == 2
    assert result.filtered_result_count == 1
    assert result.classified_result_count == 1
    assert result.document_count == 1
    assert result.evidence_document_count == 1
    assert result.evidence_chars == 5
    assert result.preferred_source_type_match is True
    assert result.step_durations["research"] >= 0


def test_summarize_evaluation_results() -> None:
    results = [
        NewsEvaluationResult(
            case_id="a",
            category="news",
            query="a",
            completed=True,
            passed_minimums=True,
            document_count=2,
            evidence_document_count=1,
            no_evidence_document_count=1,
            evidence_chars=30,
            notes_rewrite_count=2,
            notes_chars=12,
            citations=[
                Citation(
                    reference="S1",
                    document_id="document",
                    source_url="citation",
                )
            ],
        ),
        NewsEvaluationResult(case_id="b", category="news", query="b"),
    ]

    summary = summarize_evaluation_results(results)

    assert summary["case_count"] == 2
    assert summary["completed_count"] == 1
    assert summary["average_documents"] == 1.0
    assert summary["average_evidence_chars"] == 15.0
    assert summary["average_evidence_documents"] == 0.5
    assert summary["average_no_evidence_documents"] == 0.5
    assert summary["total_notes_rewrites"] == 2
    assert summary["average_notes_chars"] == 6.0


def test_evaluation_reports_handled_research_failures() -> None:
    sink = InMemoryTraceSink()
    runtime = AgentRuntime(
        Policy(AgentActionType.STOP),
        Executor(
            FailedResearchObservation(
                query="focused",
                route="web",
                stage="retrieval",
                provider="tavily",
                reason="http_status",
                status_code=429,
                message="rate limited",
                source_error_type="HTTPStatusError",
                retryable=True,
                attempt_count=2,
            )
        ),
        tracer=Tracer(sink),
    )
    output = asyncio.run(runtime.run(AgentState(query=UserQuery(text="question"))))
    result = extract_evaluation_result(
        NewsEvaluationCase(id="case", category="news", query="question"),
        output,
        InMemoryArtifactStore(),
        sink.get_trace(output.trace_id),
    )
    summary = summarize_evaluation_results([result])

    assert len(result.research_failures) == 1
    assert result.research_failures[0]["stage"] == "retrieval"
    assert result.research_failures[0]["reason"] == "http_status"
    assert summary["with_research_failures_count"] == 1
    assert summary["research_failure_stage_counts"] == {"retrieval": 1}
    assert summary["research_failure_reason_counts"] == {"http_status": 1}
