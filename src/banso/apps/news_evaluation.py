"""Models and result extraction for news runtime evaluations."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from banso.artifacts import ArtifactStore
from banso.core import RuntimeRunResult
from banso.retrieval import SearchResult


class NewsEvaluationCase(BaseModel):
    """One repeatable AI news evaluation query and its minimum expectations."""

    id: str
    category: str
    query: str
    language: str | None = None
    region: str | None = None
    time_range: str | None = None
    preferred_source_types: list[str] = Field(default_factory=list)
    min_documents: int = 1
    min_evidence: int = 1
    min_citations: int = 1
    notes: str | None = None


class NewsEvaluationResult(BaseModel):
    """Machine-readable outcome for one evaluation case."""

    case_id: str
    category: str
    query: str
    completed: bool = False
    passed_minimums: bool = False
    final_answer: str | None = None
    retrieved_result_count: int = 0
    filtered_result_count: int = 0
    admitted_result_count: int = 0
    rejected_result_count: int = 0
    source_rejections: list[dict[str, Any]] = Field(default_factory=list)
    document_count: int = 0
    evidence_count: int = 0
    citations: list[str] = Field(default_factory=list)
    source_domains: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    preferred_source_types: list[str] = Field(default_factory=list)
    preferred_source_type_match: bool = False
    document_read_failures: list[dict[str, Any]] = Field(default_factory=list)
    evidence_extraction_failures: list[dict[str, Any]] = Field(default_factory=list)
    step_durations: dict[str, float] = Field(default_factory=dict)
    total_action_seconds: float = 0.0
    error_type: str | None = None
    error_message: str | None = None


def load_evaluation_cases(path: Path) -> list[NewsEvaluationCase]:
    """Load non-empty JSONL records from a case file."""

    cases: list[NewsEvaluationCase] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(NewsEvaluationCase.model_validate_json(line))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid evaluation case at {path}:{line_number}") from error
    return cases


def extract_evaluation_result(
    case: NewsEvaluationCase,
    output: RuntimeRunResult,
    store: ArtifactStore,
) -> NewsEvaluationResult:
    """Convert a runtime output into stable evaluation fields."""

    state = output.result.state
    observations = {
        step.action.type.value: step.observation for step in output.trace.steps
    }
    synthesis = observations.get("synthesize")
    search = observations.get("search")
    read_document = observations.get("read_document")
    extract_evidence = observations.get("extract_evidence")
    citations = synthesis.data.get("citations", []) if synthesis else []
    citations = [value for value in citations if isinstance(value, str)]

    sources = [
        result.source
        for result_id in state.search_result_ids
        if (result := store.get(result_id, SearchResult)) is not None
        and result.source is not None
    ]
    source_domains = list(
        dict.fromkeys(source.name for source in sources if source.name)
    )
    source_types = list(dict.fromkeys(source.type.value for source in sources))
    preferred_source_type_match = bool(
        set(case.preferred_source_types) & set(source_types)
    )
    filter_report = search.data.get("retrieval_filter_report", {}) if search else {}
    evaluation_report = (
        search.data.get("search_result_evaluation_report", {}) if search else {}
    )
    evaluations = evaluation_report.get("evaluations", [])
    source_rejections = [
        evaluation
        for evaluation in evaluations
        if isinstance(evaluation, dict) and evaluation.get("accepted") is False
    ]
    step_durations = {
        step.action.type.value: step.duration_seconds or 0.0
        for step in output.trace.steps
    }
    passed_minimums = (
        state.done
        and len(state.document_ids) >= case.min_documents
        and len(state.evidence_ids) >= case.min_evidence
        and len(citations) >= case.min_citations
        and bool(output.result.final_answer)
    )

    return NewsEvaluationResult(
        case_id=case.id,
        category=case.category,
        query=case.query,
        completed=state.done,
        passed_minimums=passed_minimums,
        final_answer=output.result.final_answer,
        retrieved_result_count=filter_report.get("input_count", 0),
        filtered_result_count=filter_report.get("output_count", 0),
        admitted_result_count=evaluation_report.get(
            "accepted_count", len(state.search_result_ids)
        ),
        rejected_result_count=evaluation_report.get("rejected_count", 0),
        source_rejections=source_rejections,
        document_count=len(state.document_ids),
        evidence_count=len(state.evidence_ids),
        citations=citations,
        source_domains=source_domains,
        source_types=source_types,
        preferred_source_types=case.preferred_source_types,
        preferred_source_type_match=preferred_source_type_match,
        document_read_failures=(
            read_document.data.get("document_read_failures", [])
            if read_document
            else []
        ),
        evidence_extraction_failures=(
            extract_evidence.data.get("evidence_extraction_failures", [])
            if extract_evidence
            else []
        ),
        step_durations=step_durations,
        total_action_seconds=sum(step_durations.values()),
    )


def summarize_evaluation_results(
    results: list[NewsEvaluationResult],
) -> dict[str, Any]:
    """Aggregate objective pipeline metrics across evaluation cases."""

    count = len(results)
    if count == 0:
        return {"case_count": 0}

    rejection_reason_counts: dict[str, int] = {}
    for result in results:
        for rejection in result.source_rejections:
            for reason in rejection.get("reasons", []):
                if isinstance(reason, str):
                    rejection_reason_counts[reason] = (
                        rejection_reason_counts.get(reason, 0) + 1
                    )

    return {
        "case_count": count,
        "completed_count": sum(result.completed for result in results),
        "passed_minimums_count": sum(result.passed_minimums for result in results),
        "with_documents_count": sum(result.document_count > 0 for result in results),
        "with_evidence_count": sum(result.evidence_count > 0 for result in results),
        "with_citations_count": sum(bool(result.citations) for result in results),
        "preferred_source_match_count": sum(
            result.preferred_source_type_match for result in results
        ),
        "error_count": sum(result.error_type is not None for result in results),
        "source_rejection_reasons": rejection_reason_counts,
        "average_retrieved_results": round(
            sum(result.retrieved_result_count for result in results) / count, 2
        ),
        "average_filtered_results": round(
            sum(result.filtered_result_count for result in results) / count, 2
        ),
        "average_admitted_results": round(
            sum(result.admitted_result_count for result in results) / count, 2
        ),
        "average_rejected_results": round(
            sum(result.rejected_result_count for result in results) / count, 2
        ),
        "average_documents": round(
            sum(result.document_count for result in results) / count, 2
        ),
        "average_evidence": round(
            sum(result.evidence_count for result in results) / count, 2
        ),
        "average_action_seconds": round(
            sum(result.total_action_seconds for result in results) / count, 2
        ),
    }
