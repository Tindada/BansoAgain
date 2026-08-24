"""Score an existing GISA results file without model or retrieval calls.

Run with:
uv run --group evaluation python scripts/score_gisa.py \
  runs/gisa/<run>/results.jsonl
"""

import argparse
import json
from pathlib import Path

from banso.benchmarks.gisa_scoring import score_gisa_results


DEFAULT_ANSWER_DIR = Path("evaluations/gisa/raw/answer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--answer-dir", type=Path, default=DEFAULT_ANSWER_DIR)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(args: argparse.Namespace) -> None:
    scores, summary = score_gisa_results(args.results, args.answer_dir)
    output_dir = args.results.parent
    scores_path = output_dir / "scores.jsonl"
    summary_path = output_dir / "score_summary.json"
    scores_path.write_text(
        "".join(score.model_dump_json() + "\n" for score in scores),
        encoding="utf-8",
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"scores: {scores_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main(parse_args())
