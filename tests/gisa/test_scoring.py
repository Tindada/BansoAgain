"""Tests for deterministic offline GISA scoring."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from banso.benchmarks.gisa import GisaAnswerType
from banso.benchmarks.gisa_scoring import (
    SimpleEvaluator,
    score_gisa_results,
)


def test_gisa_item_and_set_metrics(tmp_path: Path) -> None:
    evaluator = SimpleEvaluator()
    item_truth = tmp_path / "item.csv"
    item_truth.write_text("0.5\n", encoding="utf-8")
    assert evaluator.evaluate_one(
        "```tsv\nValue\n50%\n```",
        item_truth,
        GisaAnswerType.ITEM,
    ) == {"item_em": 1, "global_em": 1}

    set_truth = tmp_path / "set.csv"
    set_truth.write_text("A\nB\n", encoding="utf-8")
    assert evaluator.evaluate_one(
        "```tsv\nItem\nA\nC\nA\n```",
        set_truth,
        GisaAnswerType.SET,
    ) == {
        "set_precision": 0.5,
        "set_recall": 0.5,
        "set_f1": 0.5,
        "global_em": 0,
    }


def test_gisa_list_metrics_preserve_duplicates_and_order(tmp_path: Path) -> None:
    truth = tmp_path / "list.csv"
    truth.write_text("A\nB\nA\n", encoding="utf-8")

    metrics = SimpleEvaluator().evaluate_one(
        "```tsv\nItem\nA\nA\nB\n```",
        truth,
        GisaAnswerType.LIST,
    )

    assert metrics == {
        "list_content_f1": 1.0,
        "list_order_score": 0.6667,
        "global_em": 0,
    }


def test_gisa_table_metrics_use_common_columns_and_column_value_items(
    tmp_path: Path,
) -> None:
    truth = tmp_path / "table.csv"
    truth.write_text("Name,Year\nA,1\nB,2\n", encoding="utf-8")

    metrics = SimpleEvaluator().evaluate_one(
        "```tsv\nYear\tName\tExtra\n1\tA\tx\n3\tC\ty\n```",
        truth,
        GisaAnswerType.TABLE,
    )

    assert metrics == {
        "table_row_f1": 0.5,
        "table_row_precision": 0.5,
        "table_row_recall": 0.5,
        "table_item_f1": 0.4,
        "table_item_precision": pytest.approx(1 / 3),
        "table_item_recall": 0.5,
        "global_em": 0,
    }


def test_score_gisa_results_includes_missing_predictions(tmp_path: Path) -> None:
    answer_dir = tmp_path / "answers"
    answer_dir.mkdir()
    (answer_dir / "1.csv").write_text("\ufeffAnswer\n", encoding="utf-8")
    (answer_dir / "2.csv").write_text("A\nB\n", encoding="utf-8")
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "case_id": 1,
                        "answer_type": "item",
                        "prediction": "```tsv\nValue\nanswer\n```",
                    }
                ),
                json.dumps(
                    {"case_id": 2, "answer_type": "set", "prediction": None}
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    scores, summary = score_gisa_results(results_path, answer_dir)

    assert [score.prediction_present for score in scores] == [True, False]
    assert scores[1].metrics == {
        "set_precision": 0.0,
        "set_recall": 0.0,
        "set_f1": 0.0,
        "global_em": 0,
    }
    assert summary["overall_global_em"] == 0.5
    assert summary["item"] == {
        "num_samples": 1,
        "overall_item_em": 1.0,
        "overall_global_em": 1.0,
    }


def test_score_gisa_results_rejects_duplicate_ids(tmp_path: Path) -> None:
    answer_dir = tmp_path / "answers"
    answer_dir.mkdir()
    (answer_dir / "1.csv").write_text("A\n", encoding="utf-8")
    results_path = tmp_path / "results.jsonl"
    row = json.dumps({"case_id": 1, "answer_type": "item", "prediction": None})
    results_path.write_text(f"{row}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate GISA case ID"):
        score_gisa_results(results_path, answer_dir)


def test_score_gisa_cli_writes_scores_and_summary(tmp_path: Path) -> None:
    answer_dir = tmp_path / "answers"
    answer_dir.mkdir()
    (answer_dir / "1.csv").write_text("A\n", encoding="utf-8")
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        json.dumps(
            {
                "case_id": 1,
                "answer_type": "item",
                "prediction": "```tsv\nValue\nA\n```",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/score_gisa.py",
            str(results_path),
            "--answer-dir",
            str(answer_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    score = json.loads((tmp_path / "scores.jsonl").read_text(encoding="utf-8"))
    summary = json.loads(
        (tmp_path / "score_summary.json").read_text(encoding="utf-8")
    )
    assert score["metrics"] == {"item_em": 1, "global_em": 1}
    assert summary["overall_global_em"] == 1.0
