"""Tests for action-local bounded retries."""

import asyncio

import pytest

from banso.agent.executors.retry import RetryPolicy, run_with_retry


class RetryError(Exception):
    def __init__(self, retryable: bool) -> None:
        self.retryable = retryable


def test_retry_succeeds_without_exposing_intermediate_failures() -> None:
    attempt_count = 0

    async def operation() -> str:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise RetryError(retryable=True)
        return "done"

    result = asyncio.run(
        run_with_retry(
            operation,
            error_type=RetryError,
            is_retryable=lambda error: error.retryable,
            policy=RetryPolicy(max_attempts=3, delay_seconds=0),
        )
    )

    assert result.value == "done"
    assert result.error is None
    assert result.attempt_count == 3


def test_non_retryable_error_stops_after_one_attempt() -> None:
    async def operation() -> str:
        raise RetryError(retryable=False)

    result = asyncio.run(
        run_with_retry(
            operation,
            error_type=RetryError,
            is_retryable=lambda error: error.retryable,
            policy=RetryPolicy(max_attempts=3, delay_seconds=0),
        )
    )

    assert result.value is None
    assert isinstance(result.error, RetryError)
    assert result.attempt_count == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"delay_seconds": -1},
    ],
)
def test_retry_policy_rejects_invalid_limits(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)
