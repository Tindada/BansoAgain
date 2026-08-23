"""Run Banso over locally prepared GISA cases or prepare a dry-run manifest.

Dry run (no model, retrieval, or ground-truth access):
UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_gisa.py --dry-run
"""

import argparse
import asyncio
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from banso.apps.real_news import RealNewsRuntimeBundle, build_real_news_runtime
from banso.benchmarks.gisa import (
    GisaAnswerType,
    GisaCase,
    GisaPrediction,
    GisaQuestionType,
    export_gisa_predictions,
    load_gisa_cases,
    select_balanced_gisa_cases,
    select_gisa_cases,
)
from banso.benchmarks.gisa_synthesizer import GisaSynthesizer
from banso.agent.action import AgentActionType
from banso.agent.runtime import RuntimeExecutionError
from banso.agent.state import AgentState, ExecutionBudget, UserQuery
from banso.tracing.trace import SpanRecord


DEFAULT_CASES = Path("evaluations/gisa/derived/questions.jsonl")


class GisaEvaluationResult(BaseModel):
    """Machine-readable result for one attempted GISA case."""

    case_id: int
    answer_type: GisaAnswerType
    question_type: GisaQuestionType
    topic: str
    completed: bool = False
    prediction: str | None = None
    trace_id: str | None = None
    stop_reason: str | None = None
    step_count: int = 0
    research_count: int = 0
    document_count: int = 0
    evidence_count: int = 0
    total_seconds: float = 0.0
    error_type: str | None = None
    error_message: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--question-type",
        choices=[question_type.value for question_type in GisaQuestionType],
    )
    parser.add_argument(
        "--answer-type",
        action="append",
        choices=[answer_type.value for answer_type in GisaAnswerType],
        dest="answer_types",
        help="Repeat to include more than one answer type.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--per-answer-type",
        type=int,
        help="Select this many cases for each included answer type.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-researches", type=int, default=5)
    parser.add_argument("--max-results-per-research", type=int, default=10)
    parser.add_argument("--max-active-documents", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare a run manifest without invoking Banso or reading answers.",
    )
    return parser.parse_args()


def default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(f"runs/gisa/gisa_{timestamp}")


def select_cases(args: argparse.Namespace) -> list[GisaCase]:
    cases = load_gisa_cases(args.cases)
    question_type = (
        GisaQuestionType(args.question_type) if args.question_type else None
    )
    answer_types = (
        {GisaAnswerType(value) for value in args.answer_types}
        if args.answer_types
        else None
    )
    if args.per_answer_type is not None and args.limit is not None:
        raise SystemExit("--per-answer-type and --limit cannot be used together")
    if args.per_answer_type is not None:
        selected = select_balanced_gisa_cases(
            cases,
            per_answer_type=args.per_answer_type,
            question_type=question_type,
            answer_types=answer_types,
        )
    else:
        selected = select_gisa_cases(
            cases,
            question_type=question_type,
            answer_types=answer_types,
            limit=args.limit,
        )
    if not selected:
        raise SystemExit("no GISA cases matched the requested filters")
    return selected


def build_manifest(
    args: argparse.Namespace,
    cases: list[GisaCase],
) -> dict[str, object]:
    mode = "dry-run" if args.dry_run else "run"
    answer_type_counts = Counter(case.answer_type.value for case in cases)
    question_type_counts = Counter(case.question_type.value for case in cases)
    topic_counts = Counter(case.topic for case in cases)
    return {
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(args.cases),
        "selected_case_count": len(cases),
        "selected_case_ids": [case.id for case in cases],
        "filters": {
            "question_type": args.question_type,
            "answer_types": sorted(args.answer_types) if args.answer_types else None,
            "limit": args.limit,
            "per_answer_type": args.per_answer_type,
        },
        "answer_types": dict(sorted(answer_type_counts.items())),
        "question_types": dict(sorted(question_type_counts.items())),
        "topics": dict(sorted(topic_counts.items())),
        "budget": {
            "max_steps": args.max_steps,
            "max_researches": args.max_researches,
            "max_results_per_research": args.max_results_per_research,
            "max_active_documents": args.max_active_documents,
        },
        "runtime": (
            {
                "retrieval_routes": os.getenv(
                    "BANSO_NEWS_RETRIEVAL_ROUTES", "web"
                ),
                "policy_and_extraction_model": os.getenv("VLLM_MODEL"),
                "synthesis_model": os.getenv("EXTERNAL_LLM_MODEL"),
            }
            if mode == "run"
            else None
        ),
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def execution_budget(args: argparse.Namespace) -> ExecutionBudget:
    return ExecutionBudget(
        max_steps=args.max_steps,
        max_researches=args.max_researches,
        max_results_per_research=args.max_results_per_research,
        max_active_documents=args.max_active_documents,
    )


def stop_reason(state: AgentState) -> str:
    if not state.done and state.current_step >= state.budget.max_steps:
        return "max_steps"
    if state.last_action is not None:
        return state.last_action.value
    return "incomplete"


def total_trace_seconds(spans: list[SpanRecord]) -> float:
    root = next((span for span in spans if span.name == "agent.run"), None)
    return root.duration_seconds if root is not None else 0.0


async def run_case(
    case: GisaCase,
    bundle: RealNewsRuntimeBundle,
    budget: ExecutionBudget,
) -> tuple[GisaEvaluationResult, list[SpanRecord]]:
    print(f"running GISA {case.id} ({case.answer_type.value})", flush=True)
    spans: list[SpanRecord] = []
    try:
        output = await bundle.runtime.run(
            AgentState(
                query=UserQuery(text=case.question, language="en"),
                synthesis_metadata={
                    "gisa": {"answer_type": case.answer_type.value}
                },
                budget=budget,
            )
        )
        state = output.result.state
        spans = bundle.trace_sink.get_trace(output.trace_id)
        result = GisaEvaluationResult(
            case_id=case.id,
            answer_type=case.answer_type,
            question_type=case.question_type,
            topic=case.topic,
            completed=state.done,
            prediction=state.final_answer,
            trace_id=output.trace_id,
            stop_reason=stop_reason(state),
            step_count=state.current_step,
            research_count=sum(
                entry.action.type == AgentActionType.RESEARCH
                for entry in state.action_history
            ),
            document_count=len(state.documents),
            evidence_count=sum(
                len(document.evidence_ids) for document in state.documents.values()
            ),
            total_seconds=total_trace_seconds(spans),
        )
    except RuntimeExecutionError as error:
        spans = bundle.trace_sink.get_trace(error.trace_id)
        result = GisaEvaluationResult(
            case_id=case.id,
            answer_type=case.answer_type,
            question_type=case.question_type,
            topic=case.topic,
            trace_id=error.trace_id or None,
            total_seconds=total_trace_seconds(spans),
            error_type=type(error.original_error).__name__,
            error_message=str(error.original_error),
        )
    except Exception as error:
        result = GisaEvaluationResult(
            case_id=case.id,
            answer_type=case.answer_type,
            question_type=case.question_type,
            topic=case.topic,
            error_type=type(error).__name__,
            error_message=str(error),
        )
    print(
        f"finished GISA {case.id}: prediction={result.prediction is not None}, "
        f"documents={result.document_count}, evidence={result.evidence_count}",
        flush=True,
    )
    return result, spans


def summarize(results: list[GisaEvaluationResult]) -> dict[str, object]:
    return {
        "case_count": len(results),
        "completed_count": sum(result.completed for result in results),
        "prediction_count": sum(
            result.prediction is not None for result in results
        ),
        "error_count": sum(result.error_type is not None for result in results),
        "total_documents": sum(result.document_count for result in results),
        "total_evidence": sum(result.evidence_count for result in results),
        "total_seconds": sum(result.total_seconds for result in results),
    }


async def run_evaluation(
    args: argparse.Namespace,
    cases: list[GisaCase],
    output_dir: Path,
) -> None:
    bundle = build_real_news_runtime(synthesizer_class=GisaSynthesizer)
    budget = execution_budget(args)
    results_path = output_dir / "results.jsonl"
    traces_path = output_dir / "traces.jsonl"
    results: list[GisaEvaluationResult] = []
    predictions: list[GisaPrediction] = []
    with (
        results_path.open("w", encoding="utf-8") as results_file,
        traces_path.open("w", encoding="utf-8") as traces_file,
    ):
        for case in cases:
            result, spans = await run_case(case, bundle, budget)
            results.append(result)
            results_file.write(result.model_dump_json() + "\n")
            results_file.flush()
            if result.prediction is not None:
                predictions.append(
                    GisaPrediction(id=result.case_id, prediction=result.prediction)
                )
            if spans:
                traces_file.write(
                    json.dumps(
                        {
                            "case_id": case.id,
                            "trace_id": spans[0].trace_id,
                            "spans": [span.model_dump(mode="json") for span in spans],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                traces_file.flush()

    predictions_path, per_case_dir = export_gisa_predictions(
        predictions, output_dir
    )
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summarize(results))
    print(f"results: {results_path}")
    print(f"traces: {traces_path}")
    print(f"predictions: {predictions_path}")
    print(f"per-case predictions: {per_case_dir}")
    print(f"summary: {summary_path}")


def main(args: argparse.Namespace) -> None:
    cases = select_cases(args)
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True)
    if not args.dry_run:
        load_dotenv()
    manifest = build_manifest(args, cases)
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"manifest: {manifest_path}")
        return
    asyncio.run(run_evaluation(args, cases, output_dir))


if __name__ == "__main__":
    main(parse_args())
