"""Run the Banso news agent."""

import argparse
import asyncio
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

from banso.apps.real_news import build_real_news_runtime
from banso.agent.runtime import RuntimeExecutionError
from banso.agent.state import AgentState, UserQuery


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="news question to research")
    parser.add_argument("--language", help="preferred answer language")
    parser.add_argument("--region", help="geographic context for the query")
    parser.add_argument("--time-range", help="time constraint for the query")
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    bundle = build_real_news_runtime()
    output = await bundle.runtime.run(
        AgentState(
            query=UserQuery(
                text=args.query,
                language=args.language,
                region=args.region,
                time_range=args.time_range,
            )
        )
    )
    state = output.result.state
    if state.final_answer is None:
        print("No final answer was produced.", file=sys.stderr)
        return 1

    print(state.final_answer)
    if state.citations:
        print("\nSources:")
        for index, citation in enumerate(state.citations, start=1):
            print(f"{index}. {citation}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    try:
        return asyncio.run(run(args))
    except RuntimeExecutionError as error:
        print(f"Error: {error}", file=sys.stderr)
        if error.trace_id:
            print(f"Trace ID: {error.trace_id}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
