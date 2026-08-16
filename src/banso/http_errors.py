"""Shared classification for failures raised by HTTPX."""

from dataclasses import dataclass
from typing import Literal

import httpx


HTTPFailureKind = Literal["http_status", "transport"]


@dataclass(frozen=True)
class HTTPFailure:
    kind: HTTPFailureKind
    status_code: int | None
    source_error_type: str


def is_retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 425, 429} or 500 <= status_code < 600


def classify_httpx_error(
    error: httpx.HTTPStatusError | httpx.TransportError,
) -> HTTPFailure:
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        return HTTPFailure(
            kind="http_status",
            status_code=status_code,
            source_error_type=type(error).__name__,
        )
    return HTTPFailure(
        kind="transport",
        status_code=None,
        source_error_type=type(error).__name__,
    )
