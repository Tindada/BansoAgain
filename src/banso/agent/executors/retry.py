"""Bounded retry support for action-local external operations."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Execution-layer limits for one external operation."""

    max_attempts: int = 2
    delay_seconds: float = 0.1

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.delay_seconds < 0:
            raise ValueError("delay_seconds must be non-negative")


@dataclass(frozen=True)
class AttemptResult[T, E: Exception]:
    """Final value or handled error from one retried operation."""

    value: T | None
    error: E | None
    attempt_count: int


async def run_with_retry[T, E: Exception](
    operation: Callable[[], Awaitable[T]],
    *,
    error_type: type[E],
    is_retryable: Callable[[E], bool],
    policy: RetryPolicy,
) -> AttemptResult[T, E]:
    """Run an operation until success or a handled error becomes terminal."""
    for attempt_count in range(1, policy.max_attempts + 1):
        try:
            return AttemptResult(
                value=await operation(),
                error=None,
                attempt_count=attempt_count,
            )
        except error_type as error:
            if attempt_count == policy.max_attempts or not is_retryable(error):
                return AttemptResult(
                    value=None,
                    error=error,
                    attempt_count=attempt_count,
                )
            if policy.delay_seconds:
                await asyncio.sleep(policy.delay_seconds)
    raise AssertionError("retry loop completed without a result")
