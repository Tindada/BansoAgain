"""Decrypt and validate a local copy of the GISA benchmark questions.

Run with:
UV_CACHE_DIR=.uv-cache uv run python scripts/prepare_gisa.py
"""

import argparse
import base64
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("evaluations/gisa/raw/encrypted_question.jsonl")
DEFAULT_ANSWER_DIR = Path("evaluations/gisa/raw/answer")
DEFAULT_TRACE_DIR = Path("evaluations/gisa/raw/trace")
DEFAULT_OUTPUT = Path("evaluations/gisa/derived/questions.jsonl")

ANSWER_TYPES = {"item", "set", "list", "table"}
QUESTION_TYPES = {"stable", "live"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--answer-dir", type=Path, default=DEFAULT_ANSWER_DIR)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def derive_key(password: str, length: int) -> bytes:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def decrypt(ciphertext: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext, validate=True)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(value ^ key_byte for value, key_byte in zip(encrypted, key))
    return decrypted.decode("utf-8")


def load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"GISA question file not found: {path}")

    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON at {path}:{line_number}") from error
        if not isinstance(item, dict):
            raise ValueError(f"expected an object at {path}:{line_number}")

        required = {
            "id",
            "question",
            "answer_type",
            "question_type",
            "topic",
            "canary",
        }
        missing = required - item.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"missing fields at {path}:{line_number}: {names}")

        question_id = str(item["id"])
        if question_id in seen_ids:
            raise ValueError(f"duplicate question id at {path}:{line_number}: {question_id}")
        seen_ids.add(question_id)

        answer_type = str(item["answer_type"]).strip().casefold()
        if answer_type not in ANSWER_TYPES:
            raise ValueError(
                f"unsupported answer_type at {path}:{line_number}: {answer_type}"
            )
        question_type = str(item["question_type"]).strip().casefold()
        if question_type not in QUESTION_TYPES:
            raise ValueError(
                f"unsupported question_type at {path}:{line_number}: {question_type}"
            )

        try:
            plaintext = decrypt(str(item["question"]), str(item["canary"])).strip()
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(
                f"could not decrypt question at {path}:{line_number}"
            ) from error
        if not plaintext:
            raise ValueError(f"empty decrypted question at {path}:{line_number}")

        questions.append(
            {
                "id": item["id"],
                "question": plaintext,
                "answer_type": answer_type,
                "question_type": question_type,
                "topic": str(item["topic"]).strip(),
            }
        )

    if not questions:
        raise ValueError(f"no questions found in {path}")
    return questions


def file_ids(directory: Path, suffix: str) -> set[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"GISA asset directory not found: {directory}")
    return {path.stem for path in directory.iterdir() if path.suffix == suffix}


def validate_assets(
    questions: list[dict[str, Any]], answer_dir: Path, trace_dir: Path
) -> None:
    question_ids = {str(question["id"]) for question in questions}
    for label, actual_ids in (
        ("answer", file_ids(answer_dir, ".csv")),
        ("trace", file_ids(trace_dir, ".json")),
    ):
        missing = sorted(question_ids - actual_ids)
        extra = sorted(actual_ids - question_ids)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing {label} IDs: {', '.join(missing[:10])}")
            if extra:
                details.append(f"extra {label} IDs: {', '.join(extra[:10])}")
            raise ValueError("; ".join(details))


def write_questions(path: Path, questions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(question, ensure_ascii=False) + "\n" for question in questions
    )
    path.write_text(content, encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    questions = load_questions(args.input)
    validate_assets(questions, args.answer_dir, args.trace_dir)
    write_questions(args.output, questions)

    answer_counts = Counter(question["answer_type"] for question in questions)
    question_counts = Counter(question["question_type"] for question in questions)
    topic_counts = Counter(question["topic"] for question in questions)
    summary = {
        "question_count": len(questions),
        "answer_types": dict(sorted(answer_counts.items())),
        "question_types": dict(sorted(question_counts.items())),
        "topics": dict(sorted(topic_counts.items())),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(parse_args())
