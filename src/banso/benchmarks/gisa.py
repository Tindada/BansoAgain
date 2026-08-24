"""GISA case loading and official prediction-format adapters."""

import csv
import json
from collections.abc import Iterable, Sequence
from enum import StrEnum
from io import StringIO
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class GisaAnswerType(StrEnum):
    """Structured answer types defined by GISA."""

    ITEM = "item"
    SET = "set"
    LIST = "list"
    TABLE = "table"


class GisaQuestionType(StrEnum):
    """Static and periodically refreshed GISA subsets."""

    STABLE = "stable"
    LIVE = "live"


class GisaCase(BaseModel):
    """One locally decrypted GISA case without ground-truth data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(ge=0)
    question: str = Field(min_length=1)
    answer_type: GisaAnswerType
    question_type: GisaQuestionType
    topic: str = Field(min_length=1)

    @field_validator("question", "topic")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value


class GisaPrediction(BaseModel):
    """One prediction accepted by the current GISA leaderboard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(ge=0)
    prediction: str = Field(min_length=1)

    @field_validator("prediction")
    @classmethod
    def require_tsv_block(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith("```tsv\n") or not value.endswith("```"):
            raise ValueError("prediction must be enclosed in a TSV code block")
        return value


def load_gisa_cases(path: Path) -> list[GisaCase]:
    """Load locally decrypted questions without accessing answers or traces."""

    if not path.is_file():
        raise FileNotFoundError(f"GISA case file not found: {path}")

    cases: list[GisaCase] = []
    seen_ids: set[int] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            case = GisaCase.model_validate_json(line)
        except (ValidationError, ValueError) as error:
            raise ValueError(f"invalid GISA case at {path}:{line_number}") from error
        if case.id in seen_ids:
            raise ValueError(f"duplicate GISA case ID at {path}:{line_number}: {case.id}")
        seen_ids.add(case.id)
        cases.append(case)

    if not cases:
        raise ValueError(f"no GISA cases found in {path}")
    return cases


def select_gisa_cases(
    cases: Iterable[GisaCase],
    *,
    question_type: GisaQuestionType | None = None,
    answer_types: set[GisaAnswerType] | None = None,
    case_ids: set[int] | None = None,
    limit: int | None = None,
) -> list[GisaCase]:
    """Apply deterministic subset filters in source-file order."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")
    cases = list(cases)
    if case_ids is not None:
        unknown_ids = case_ids - {case.id for case in cases}
        if unknown_ids:
            values = ", ".join(str(case_id) for case_id in sorted(unknown_ids))
            raise ValueError(f"unknown GISA case IDs: {values}")

    selected = [
        case
        for case in cases
        if (case_ids is None or case.id in case_ids)
        and (question_type is None or case.question_type == question_type)
        and (answer_types is None or case.answer_type in answer_types)
    ]
    return selected[:limit] if limit is not None else selected


def select_balanced_gisa_cases(
    cases: Iterable[GisaCase],
    *,
    per_answer_type: int,
    question_type: GisaQuestionType | None = None,
    answer_types: set[GisaAnswerType] | None = None,
) -> list[GisaCase]:
    """Select the same number of source-ordered cases for each answer type."""

    if per_answer_type < 1:
        raise ValueError("per_answer_type must be at least 1")
    included_types = answer_types or set(GisaAnswerType)
    buckets = {answer_type: [] for answer_type in included_types}
    for case in cases:
        if question_type is not None and case.question_type != question_type:
            continue
        if case.answer_type not in buckets:
            continue
        bucket = buckets[case.answer_type]
        if len(bucket) < per_answer_type:
            bucket.append(case)

    if any(len(bucket) < per_answer_type for bucket in buckets.values()):
        raise ValueError("not enough matching GISA cases for a balanced selection")

    selected: list[GisaCase] = []
    for answer_type in GisaAnswerType:
        selected.extend(buckets.get(answer_type, []))
    return selected


def render_gisa_tsv(
    answer_type: GisaAnswerType,
    rows: Sequence[Sequence[object]],
    *,
    columns: Sequence[str] | None = None,
) -> str:
    """Render a structured answer as the TSV block expected by GISA."""

    if answer_type == GisaAnswerType.ITEM:
        headers = ["Value"]
        if len(rows) != 1 or len(rows[0]) != 1:
            raise ValueError("item answers require exactly one row and one value")
    elif answer_type in {GisaAnswerType.SET, GisaAnswerType.LIST}:
        headers = ["Item"]
        if any(len(row) != 1 for row in rows):
            raise ValueError("set and list answers require one value per row")
    else:
        if not columns or any(not str(column).strip() for column in columns):
            raise ValueError("table answers require non-blank column names")
        headers = [str(column).strip() for column in columns]
        if len(set(headers)) != len(headers):
            raise ValueError("table column names must be unique")
        if any(len(row) != len(headers) for row in rows):
            raise ValueError("every table row must match the number of columns")

    buffer = StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(
        ["" if value is None else str(value) for value in row] for row in rows
    )
    return f"```tsv\n{buffer.getvalue()}```"


def export_gisa_predictions(
    predictions: Sequence[GisaPrediction], output_dir: Path
) -> tuple[Path, Path]:
    """Write both leaderboard JSONL and local-evaluator per-case JSON files."""

    prediction_ids = [prediction.id for prediction in predictions]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise ValueError("GISA predictions must contain unique IDs")

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "predictions.jsonl"
    per_case_dir = output_dir / "predictions"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path.write_text(
        "".join(prediction.model_dump_json() + "\n" for prediction in predictions),
        encoding="utf-8",
    )
    for prediction in predictions:
        (per_case_dir / f"{prediction.id}.json").write_text(
            json.dumps(
                {"prediction": prediction.prediction},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return jsonl_path, per_case_dir
