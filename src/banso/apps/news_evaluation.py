"""Models and result extraction for news runtime evaluations."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from banso.artifacts.store import ArtifactStore
from banso.agent.action import AgentAction
from banso.agent.observation import (
    CompletedResearchObservation,
    ExtractionFailure,
    FailedResearchObservation,
    FetchFailure,
    Observation,
    validate_observation,
)
from banso.agent.runtime import RuntimeRunResult
from banso.documents.models import DocumentEvidence
from banso.retrieval.models import SearchResult
from banso.synthesis.synthesizer import Citation
from banso.tracing.trace import SpanRecord


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
    trace_id: str | None = None
    retrieved_result_count: int = 0
    filtered_result_count: int = 0
    classified_result_count: int = 0
    recognized_source_count: int = 0
    unknown_source_count: int = 0
    classification_coverage: float = 0.0
    source_classifications: list[dict[str, Any]] = Field(default_factory=list)
    document_count: int = 0
    evidence_document_count: int = 0
    no_evidence_document_count: int = 0
    evidence_chars: int = 0
    citations: list[Citation] = Field(default_factory=list)
    source_domains: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)
    preferred_source_types: list[str] = Field(default_factory=list)
    preferred_source_type_match: bool = False
    document_fetch_failures: list[dict[str, Any]] = Field(default_factory=list)
    evidence_extraction_failures: list[dict[str, Any]] = Field(default_factory=list)
    research_failures: list[dict[str, Any]] = Field(default_factory=list)
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
    spans: Sequence[SpanRecord],
) -> NewsEvaluationResult:
    """Convert a runtime output into stable evaluation fields."""

    state = output.result.state
    evidence_chars = 0
    for document in state.documents.values():
        if document.evidence_id is None:
            continue
        evidence = store.get(document.evidence_id, DocumentEvidence)
        if evidence is None:
            raise ValueError(f"Invalid DocumentEvidence: {document.evidence_id}")
        evidence_chars += len(evidence.text)
    evidence_document_count = state.evidence_document_count
    steps = _completed_steps(spans)
    document_fetch_failures = [
        {
            "search_result_id": outcome.search_result_id,
            **outcome.failure.model_dump(mode="json"),
        }
        for _, observation in steps
        if isinstance(observation, CompletedResearchObservation)
        for outcome in observation.fetch_outcomes
        if isinstance(outcome, FetchFailure)
    ]
    evidence_extraction_failures = [
        {
            "document_id": outcome.document_id,
            **outcome.failure.model_dump(mode="json"),
        }
        for _, observation in steps
        if isinstance(observation, CompletedResearchObservation)
        for outcome in observation.extraction_outcomes
        if isinstance(outcome, ExtractionFailure)
    ]
    research_failures = [
        observation.model_dump(
            mode="json",
            exclude={"type", "status", "query", "route"},
            exclude_none=True,
        )
        for _, observation in steps
        if isinstance(observation, FailedResearchObservation)
    ]

    sources = [
        result.source
        for result_id in state.search_results
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
    retrieved_result_count = 0
    filtered_result_count = 0
    classified_result_count = 0
    recognized_source_count = 0
    unknown_source_count = 0
    source_classifications: list[dict[str, Any]] = []
    for _, observation in steps:
        if not isinstance(observation, CompletedResearchObservation):
            continue
        filter_report = observation.retrieval_filter_report
        classification_report = observation.source_classification_report
        retrieved_result_count += filter_report.input_count
        filtered_result_count += filter_report.output_count
        classified_result_count += classification_report.input_count
        recognized_source_count += classification_report.recognized_count
        unknown_source_count += classification_report.unknown_count
        source_classifications.extend(
            classification.model_dump(mode="json")
            for classification in classification_report.classifications
        )
    step_durations: dict[str, float] = {}
    total_action_seconds = 0.0
    for span in spans:
        if span.name != "agent.action.execute" or span.status != "ok":
            continue
        action_type = span.attributes.get("action_type")
        if not isinstance(action_type, str):
            continue
        duration = span.duration_seconds
        step_durations[action_type] = step_durations.get(action_type, 0.0) + duration
        total_action_seconds += duration
    passed_minimums = (
        state.done
        and evidence_document_count >= case.min_documents
        and len(state.citations) >= case.min_citations
        and bool(state.final_answer)
    )

    return NewsEvaluationResult(
        case_id=case.id,
        category=case.category,
        query=case.query,
        completed=state.done,
        passed_minimums=passed_minimums,
        final_answer=state.final_answer,
        trace_id=output.trace_id,
        retrieved_result_count=retrieved_result_count,
        filtered_result_count=filtered_result_count,
        classified_result_count=classified_result_count,
        recognized_source_count=recognized_source_count,
        unknown_source_count=unknown_source_count,
        classification_coverage=_ratio(
            recognized_source_count, classified_result_count
        ),
        source_classifications=source_classifications,
        document_count=len(state.documents),
        evidence_document_count=evidence_document_count,
        no_evidence_document_count=len(state.documents) - evidence_document_count,
        evidence_chars=evidence_chars,
        citations=state.citations,
        source_domains=source_domains,
        source_types=source_types,
        preferred_source_types=case.preferred_source_types,
        preferred_source_type_match=preferred_source_type_match,
        document_fetch_failures=document_fetch_failures,
        evidence_extraction_failures=evidence_extraction_failures,
        research_failures=research_failures,
        step_durations=step_durations,
        total_action_seconds=total_action_seconds,
    )


def _completed_steps(
    spans: Sequence[SpanRecord],
) -> list[tuple[AgentAction, Observation]]:
    """Decode completed runtime steps without coupling tracing to core models."""

    decoded: list[tuple[int, AgentAction, Observation]] = []
    for span in spans:
        if span.name != "agent.step" or span.status != "ok":
            continue
        if not isinstance(span.output, dict):
            continue
        action_value = span.output.get("action")
        observation_value = span.output.get("observation")
        if action_value is None or observation_value is None:
            continue
        action = AgentAction.model_validate(action_value)
        observation = validate_observation(observation_value)
        step_index = span.attributes.get("step_index", 0)
        decoded.append(
            (
                step_index if isinstance(step_index, int) else 0,
                action,
                observation,
            )
        )
    decoded.sort(key=lambda item: item[0])
    return [(action, observation) for _, action, observation in decoded]


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def summarize_evaluation_results(results: list[NewsEvaluationResult]) -> dict[str, Any]:
    """Aggregate objective pipeline metrics across evaluation cases."""

    count = len(results)
    if count == 0:
        return {"case_count": 0}

    classification_source_counts: dict[str, int] = {}
    source_type_counts: dict[str, int] = {}
    unknown_domains: dict[str, dict[str, Any]] = {}
    for result in results:
        for classification in result.source_classifications:
            classification_source = classification.get("classification_source")
            if isinstance(classification_source, str):
                classification_source_counts[classification_source] = (
                    classification_source_counts.get(classification_source, 0) + 1
                )
            source_type = classification.get("source_type")
            if isinstance(source_type, str):
                source_type_counts[source_type] = (
                    source_type_counts.get(source_type, 0) + 1
                )
            if source_type != "unknown":
                continue
            domain = classification.get("publisher_domain")
            if not isinstance(domain, str) or not domain:
                continue
            candidate = unknown_domains.setdefault(
                domain,
                {
                    "publisher_domain": domain,
                    "count": 0,
                },
            )
            candidate["count"] += 1

    unknown_source_candidates = list(unknown_domains.values())
    unknown_source_candidates.sort(
        key=lambda item: (-item["count"], item["publisher_domain"])
    )

    total_filtered = sum(result.filtered_result_count for result in results)
    total_classified = sum(result.classified_result_count for result in results)
    total_recognized = sum(result.recognized_source_count for result in results)
    total_unknown = sum(result.unknown_source_count for result in results)
    research_failure_stage_counts: dict[str, int] = {}
    research_failure_reason_counts: dict[str, int] = {}
    for result in results:
        for failure in result.research_failures:
            stage = failure.get("stage")
            if isinstance(stage, str):
                research_failure_stage_counts[stage] = (
                    research_failure_stage_counts.get(stage, 0) + 1
                )
            reason = failure.get("reason")
            if isinstance(reason, str):
                research_failure_reason_counts[reason] = (
                    research_failure_reason_counts.get(reason, 0) + 1
                )

    return {
        "case_count": count,
        "completed_count": sum(result.completed for result in results),
        "passed_minimums_count": sum(result.passed_minimums for result in results),
        "with_documents_count": sum(result.document_count > 0 for result in results),
        "with_evidence_documents_count": sum(
            result.evidence_document_count > 0 for result in results
        ),
        "with_no_evidence_documents_count": sum(
            result.no_evidence_document_count > 0 for result in results
        ),
        "with_evidence_text_count": sum(
            result.evidence_chars > 0 for result in results
        ),
        "with_citations_count": sum(bool(result.citations) for result in results),
        "preferred_source_match_count": sum(
            result.preferred_source_type_match for result in results
        ),
        "error_count": sum(result.error_type is not None for result in results),
        "with_research_failures_count": sum(
            bool(result.research_failures) for result in results
        ),
        "research_failure_stage_counts": research_failure_stage_counts,
        "research_failure_reason_counts": research_failure_reason_counts,
        "classification_source_counts": classification_source_counts,
        "source_type_counts": source_type_counts,
        "unknown_source_candidates": unknown_source_candidates,
        "total_filtered_results": total_filtered,
        "total_classified_results": total_classified,
        "total_recognized_sources": total_recognized,
        "total_unknown_sources": total_unknown,
        "classification_coverage": _ratio(total_recognized, total_classified),
        "average_retrieved_results": round(
            sum(result.retrieved_result_count for result in results) / count, 2
        ),
        "average_filtered_results": round(
            sum(result.filtered_result_count for result in results) / count, 2
        ),
        "average_classified_results": round(
            sum(result.classified_result_count for result in results) / count, 2
        ),
        "average_recognized_sources": round(
            sum(result.recognized_source_count for result in results) / count, 2
        ),
        "average_unknown_sources": round(
            sum(result.unknown_source_count for result in results) / count, 2
        ),
        "average_documents": round(
            sum(result.document_count for result in results) / count, 2
        ),
        "average_evidence_documents": round(
            sum(result.evidence_document_count for result in results) / count, 2
        ),
        "average_no_evidence_documents": round(
            sum(result.no_evidence_document_count for result in results) / count, 2
        ),
        "average_evidence_chars": round(
            sum(result.evidence_chars for result in results) / count, 2
        ),
        "average_action_seconds": round(
            sum(result.total_action_seconds for result in results) / count, 2
        ),
    }
