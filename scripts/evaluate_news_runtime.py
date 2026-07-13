"""Run the real news runtime over a JSONL evaluation set.

Run with:
UV_CACHE_DIR=.uv-cache uv run python scripts/evaluate_news_runtime.py
"""

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from banso.apps.news_evaluation import (
    NewsEvaluationResult,
    extract_evaluation_result,
    load_evaluation_cases,
    summarize_evaluation_results,
)
from banso.apps.real_news import build_real_news_runtime
from banso.core import AgentState, ExecutionBudget, UserQuery


DEFAULT_CASES = Path("evaluations/ai_professional_news.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-documents-to-read", type=int, default=6)
    return parser.parse_args()


async def run_case(case, *, max_documents_to_read: int) -> NewsEvaluationResult:
    print(f"running {case.id}: {case.query}", flush=True)
    try:
        bundle = build_real_news_runtime()
        output = await bundle.runtime.run(
            AgentState(
                query=UserQuery(
                    text=case.query,
                    language=case.language,
                    region=case.region,
                    time_range=case.time_range,
                ),
                budget=ExecutionBudget(
                    max_documents_to_read=max_documents_to_read,
                ),
            )
        )
        result = extract_evaluation_result(case, output, bundle.store)
    except Exception as error:
        result = NewsEvaluationResult(
            case_id=case.id,
            category=case.category,
            query=case.query,
            error_type=type(error).__name__,
            error_message=str(error),
        )
    print(
        f"finished {case.id}: documents={result.document_count}, "
        f"evidence={result.evidence_count}, citations={len(result.citations)}, "
        f"passed={result.passed_minimums}",
        flush=True,
    )
    return result


async def main(args: argparse.Namespace) -> None:
    load_dotenv()
    started_at = datetime.now(timezone.utc)
    cases = load_evaluation_cases(args.cases)
    if args.limit is not None:
        cases = cases[: args.limit]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or Path(f"runs/news_evaluation_{timestamp}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[NewsEvaluationResult] = []
    with output_path.open("w", encoding="utf-8") as output_file:
        for case in cases:
            result = await run_case(
                case,
                max_documents_to_read=args.max_documents_to_read,
            )
            results.append(result)
            output_file.write(result.model_dump_json() + "\n")
            output_file.flush()

    summary = summarize_evaluation_results(results)
    summary.update(
        {
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "cases_path": str(args.cases),
            "results_path": str(output_path),
            "max_documents_to_read": args.max_documents_to_read,
            "vllm_model": os.getenv("VLLM_MODEL"),
            "external_llm_model": os.getenv("EXTERNAL_LLM_MODEL"),
        }
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"results: {output_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
