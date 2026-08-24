"""Deterministic offline scoring for GISA predictions.

``SimpleEvaluator`` is adapted from the official GISA evaluator, licensed
under Apache-2.0:
https://github.com/RUC-NLPIR/GISA/blob/main/eval_script/run_evaluation.py
"""

import difflib
import json
import re
from collections import Counter
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from banso.benchmarks.gisa import GisaAnswerType


class GisaCaseScore(BaseModel):
    """Official metrics for one GISA result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: int = Field(ge=0)
    answer_type: GisaAnswerType
    prediction_present: bool
    metrics: dict[str, int | float]


class SimpleEvaluator:
    """Official GISA normalization and metric implementation."""

    @staticmethod
    def _normalize_value(value: object) -> str:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return ""

        number = text.replace(",", "").replace("$", "")
        is_percent = number.endswith("%")
        if is_percent:
            number = number[:-1]
        try:
            numeric_value = float(number)
        except ValueError:
            return text.lower().replace(" ", "").replace("*", "").replace("\n", "")

        if is_percent:
            numeric_value /= 100.0
        if numeric_value.is_integer():
            return str(int(numeric_value))
        normalized = f"{numeric_value:.6f}".rstrip("0").rstrip(".")
        return normalized or "0"

    @classmethod
    def _normalize_frame(cls, frame: pd.DataFrame) -> pd.DataFrame:
        frame.columns = [
            str(column).strip().lower().replace(" ", "")
            for column in frame.columns
        ]
        return frame.map(cls._normalize_value)

    @classmethod
    def _extract_prediction(cls, prediction: str | None) -> pd.DataFrame | None:
        if prediction is None or not prediction.strip():
            return None
        match = re.search(r"```(?:tsv)?\s*(.*?)```", prediction, re.DOTALL)
        content = match.group(1) if match else prediction
        content = "\n".join(
            line for line in content.split("\n") if line.strip()
        )
        if not content:
            return None
        try:
            frame = pd.read_csv(StringIO(content), sep="\t")
        except Exception:
            return None
        return cls._normalize_frame(frame)

    @classmethod
    def _load_ground_truth(
        cls, path: Path, answer_type: GisaAnswerType
    ) -> pd.DataFrame:
        if not path.is_file():
            raise FileNotFoundError(f"GISA ground truth not found: {path}")
        header = "infer" if answer_type == GisaAnswerType.TABLE else None
        try:
            frame = pd.read_csv(path, header=header)
        except UnicodeDecodeError:
            frame = pd.read_csv(path, header=header, encoding="gbk")
        return cls._normalize_frame(frame)

    @staticmethod
    def _precision_recall_f1(
        true_positives: int,
        prediction_count: int,
        ground_truth_count: int,
    ) -> tuple[float, float, float]:
        precision = true_positives / prediction_count if prediction_count else 0.0
        recall = true_positives / ground_truth_count if ground_truth_count else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return precision, recall, f1

    @staticmethod
    def _flatten_table(frame: pd.DataFrame) -> list[tuple[object, object]]:
        return [
            (column, value)
            for column in frame.columns
            for value in frame[column]
        ]

    @staticmethod
    def _evaluate_item(
        prediction: pd.DataFrame | None, truth: pd.DataFrame
    ) -> dict[str, int]:
        if prediction is None or prediction.empty:
            return {"item_em": 0}
        predicted_item = "".join(prediction.iloc[0, :].tolist())
        truth_item = "".join(truth.iloc[0, :].tolist())
        return {"item_em": int(predicted_item == truth_item)}

    @classmethod
    def _evaluate_set(
        cls, prediction: pd.DataFrame | None, truth: pd.DataFrame
    ) -> dict[str, float]:
        if prediction is None or prediction.empty:
            return {"set_precision": 0.0, "set_recall": 0.0, "set_f1": 0.0}
        predicted_items = set(prediction.iloc[:, -1].tolist())
        truth_items = set(truth.iloc[:, -1].tolist())
        precision, recall, f1 = cls._precision_recall_f1(
            len(predicted_items & truth_items),
            len(predicted_items),
            len(truth_items),
        )
        return {"set_precision": precision, "set_recall": recall, "set_f1": f1}

    @classmethod
    def _evaluate_list(
        cls, prediction: pd.DataFrame | None, truth: pd.DataFrame
    ) -> dict[str, float]:
        if prediction is None or prediction.empty:
            return {"list_content_f1": 0.0, "list_order_score": 0.0}
        predicted_items = prediction.iloc[:, -1].tolist()
        truth_items = truth.iloc[:, -1].tolist()
        common_count = sum(
            (Counter(predicted_items) & Counter(truth_items)).values()
        )
        _, _, content_f1 = cls._precision_recall_f1(
            common_count, len(predicted_items), len(truth_items)
        )
        order_score = difflib.SequenceMatcher(
            None, truth_items, predicted_items
        ).ratio()
        return {
            "list_content_f1": round(content_f1, 4),
            "list_order_score": round(order_score, 4),
        }

    @classmethod
    def _evaluate_table(
        cls, prediction: pd.DataFrame | None, truth: pd.DataFrame
    ) -> dict[str, float]:
        default = {
            "table_row_f1": 0.0,
            "table_row_precision": 0.0,
            "table_row_recall": 0.0,
            "table_item_f1": 0.0,
            "table_item_precision": 0.0,
            "table_item_recall": 0.0,
        }
        if prediction is None or prediction.empty:
            return default

        common_columns = [
            column for column in truth.columns if column in prediction.columns
        ]
        if common_columns:
            predicted_rows = set(
                tuple(row)
                for row in prediction[common_columns].astype(str).to_numpy()
            )
            truth_rows = set(
                tuple(row) for row in truth[common_columns].astype(str).to_numpy()
            )
            row_precision, row_recall, row_f1 = cls._precision_recall_f1(
                len(predicted_rows & truth_rows),
                len(predicted_rows),
                len(truth_rows),
            )
        else:
            row_precision = row_recall = row_f1 = 0.0

        predicted_items = Counter(cls._flatten_table(prediction))
        truth_items = Counter(cls._flatten_table(truth))
        item_precision, item_recall, item_f1 = cls._precision_recall_f1(
            sum((predicted_items & truth_items).values()),
            sum(predicted_items.values()),
            sum(truth_items.values()),
        )
        return {
            "table_row_f1": row_f1,
            "table_row_precision": row_precision,
            "table_row_recall": row_recall,
            "table_item_f1": item_f1,
            "table_item_precision": item_precision,
            "table_item_recall": item_recall,
        }

    def evaluate_one(
        self,
        prediction: str | None,
        ground_truth_path: Path,
        answer_type: GisaAnswerType,
    ) -> dict[str, int | float]:
        """Evaluate one prediction against its official answer CSV."""

        predicted_frame = self._extract_prediction(prediction)
        truth_frame = self._load_ground_truth(ground_truth_path, answer_type)
        if answer_type == GisaAnswerType.ITEM:
            metrics: dict[str, int | float] = self._evaluate_item(
                predicted_frame, truth_frame
            )
        elif answer_type == GisaAnswerType.SET:
            metrics = self._evaluate_set(predicted_frame, truth_frame)
        elif answer_type == GisaAnswerType.LIST:
            metrics = self._evaluate_list(predicted_frame, truth_frame)
        else:
            metrics = self._evaluate_table(predicted_frame, truth_frame)

        if predicted_frame is None:
            global_em = 0
        elif answer_type == GisaAnswerType.SET:
            global_em = int(
                set(predicted_frame.iloc[:, 0].tolist())
                == set(truth_frame.iloc[:, 0].tolist())
            )
        else:
            global_em = int(
                predicted_frame.to_numpy().tolist()
                == truth_frame.to_numpy().tolist()
            )
        metrics["global_em"] = global_em
        return metrics


def summarize_gisa_scores(scores: Sequence[GisaCaseScore]) -> dict[str, object]:
    """Macro-average official metrics overall and by answer type."""

    if not scores:
        raise ValueError("no GISA scores to summarize")
    summary: dict[str, object] = {
        "overall_global_em": sum(score.metrics["global_em"] for score in scores)
        / len(scores)
    }
    for answer_type in GisaAnswerType:
        matching = [score for score in scores if score.answer_type == answer_type]
        if not matching:
            continue
        summary[answer_type.value] = {
            "num_samples": len(matching),
            **{
                f"overall_{name}": round(
                    sum(score.metrics[name] for score in matching) / len(matching), 4
                )
                for name in matching[0].metrics
            },
        }
    return summary


def score_gisa_results(
    results_path: Path, answer_dir: Path
) -> tuple[list[GisaCaseScore], dict[str, object]]:
    """Score every case recorded in a GISA ``results.jsonl`` file."""

    if not results_path.is_file():
        raise FileNotFoundError(f"GISA results not found: {results_path}")
    evaluator = SimpleEvaluator()
    scores: list[GisaCaseScore] = []
    seen_ids: set[int] = set()
    for line_number, line in enumerate(
        results_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            case_id = item["case_id"]
            if isinstance(case_id, bool) or not isinstance(case_id, int) or case_id < 0:
                raise ValueError("case_id must be a non-negative integer")
            answer_type = GisaAnswerType(item["answer_type"])
            prediction = item.get("prediction")
            if prediction is not None and not isinstance(prediction, str):
                raise ValueError("prediction must be a string or null")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid GISA result at {results_path}:{line_number}"
            ) from error
        if case_id in seen_ids:
            raise ValueError(
                f"duplicate GISA case ID at {results_path}:{line_number}: {case_id}"
            )
        seen_ids.add(case_id)

        scores.append(
            GisaCaseScore(
                case_id=case_id,
                answer_type=answer_type,
                prediction_present=bool(prediction and prediction.strip()),
                metrics=evaluator.evaluate_one(
                    prediction,
                    answer_dir / f"{case_id}.csv",
                    answer_type,
                ),
            )
        )

    if not scores:
        raise ValueError(f"no GISA results found in {results_path}")
    return scores, summarize_gisa_scores(scores)
