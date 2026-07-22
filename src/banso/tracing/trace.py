"""Business-independent tracing primitives."""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from threading import Lock
from time import perf_counter
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python


class SpanError(BaseModel):
    """An exception observed while a span was active."""

    error_type: str
    message: str


class SpanRecord(BaseModel):
    """A snapshot of one completed operation."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str
    started_at: datetime
    ended_at: datetime
    duration_seconds: float
    status: Literal["ok", "error"]
    input: Any | None = None
    output: Any | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    error: SpanError | None = None


class TraceSink(Protocol):
    """Receives completed span records."""

    def write(self, span: SpanRecord) -> None:
        """Store one completed span."""
        ...


class InMemoryTraceSink:
    """Task-safe in-memory storage for local tracing and tests."""

    def __init__(self) -> None:
        self._spans: list[SpanRecord] = []
        self._lock = Lock()

    def write(self, span: SpanRecord) -> None:
        with self._lock:
            self._spans.append(span.model_copy(deep=True))

    def get_trace(self, trace_id: str) -> list[SpanRecord]:
        with self._lock:
            spans = [
                span.model_copy(deep=True)
                for span in self._spans
                if span.trace_id == trace_id
            ]
        return sorted(spans, key=lambda span: (span.started_at, span.span_id))

    def clear(self) -> None:
        with self._lock:
            self._spans.clear()


def _trace_value(value: Any) -> Any:
    """Take a JSON-compatible snapshot without leaking tracing failures."""

    try:
        return to_jsonable_python(
            value,
            fallback=lambda item: {
                "unserializable_type": type(item).__name__,
            },
        )
    except Exception as error:
        return {
            "serialization_error": type(error).__name__,
            "value_type": type(value).__name__,
        }


class Span:
    """Mutable handle used while an operation is running."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None,
        name: str,
        input: Any | None,
        attributes: Mapping[str, Any] | None,
    ) -> None:
        self._tracer = tracer
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.name = name
        self.input = _trace_value(input) if input is not None else None
        self.output: Any | None = None
        self.attributes = _trace_value(dict(attributes or {}))
        self.started_at = datetime.now(timezone.utc)
        self._started_counter = perf_counter()

    def set_output(self, output: Any) -> None:
        self.output = _trace_value(output)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = _trace_value(value)

    def _finish(self, error: BaseException | None) -> SpanRecord:
        ended_at = datetime.now(timezone.utc)
        return SpanRecord(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            started_at=self.started_at,
            ended_at=ended_at,
            duration_seconds=perf_counter() - self._started_counter,
            status="error" if error is not None else "ok",
            input=self.input,
            output=self.output,
            attributes=self.attributes,
            error=(
                SpanError(error_type=type(error).__name__, message=str(error))
                if error is not None
                else None
            ),
        )


class _NonRecordingSpan:
    """No-op span returned when no trace is active."""

    trace_id: str | None = None
    span_id: str | None = None
    parent_span_id: str | None = None

    def set_output(self, output: Any) -> None:
        del output

    def set_attribute(self, key: str, value: Any) -> None:
        del key, value


_CURRENT_SPAN: ContextVar[Span | None] = ContextVar(
    "banso_current_span",
    default=None,
)


class Tracer:
    """Creates spans and propagates their context through the call stack."""

    def __init__(self, sink: TraceSink | None = None) -> None:
        self.sink = sink

    @contextmanager
    def start_span(
        self,
        name: str,
        *,
        input: Any | None = None,
        attributes: Mapping[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Start a child of the active span, or a new root if none is active."""

        parent = _CURRENT_SPAN.get()
        trace_id = parent.trace_id if parent is not None else uuid4().hex
        parent_span_id = parent.span_id if parent is not None else None
        with self._span_scope(
            name,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            input=input,
            attributes=attributes,
        ) as span:
            yield span

    @contextmanager
    def _span_scope(
        self,
        name: str,
        *,
        trace_id: str,
        parent_span_id: str | None,
        input: Any | None,
        attributes: Mapping[str, Any] | None,
    ) -> Generator[Span, None, None]:
        span = Span(
            tracer=self,
            trace_id=trace_id,
            span_id=uuid4().hex[:16],
            parent_span_id=parent_span_id,
            name=name,
            input=input,
            attributes=attributes,
        )
        token = _CURRENT_SPAN.set(span)
        observed_error: BaseException | None = None
        try:
            yield span
        except BaseException as error:
            observed_error = error
            raise
        finally:
            _CURRENT_SPAN.reset(token)
            self._write(span, observed_error)

    def _write(self, span: Span, error: BaseException | None) -> None:
        if self.sink is None:
            return
        try:
            self.sink.write(span._finish(error))
        except Exception:
            # Observability must never change business execution semantics.
            return


@contextmanager
def start_span(
    name: str,
    *,
    input: Any | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Generator[Span | _NonRecordingSpan, None, None]:
    """Start a child span using the tracer stored in the current context."""

    current = _CURRENT_SPAN.get()
    if current is None:
        yield _NonRecordingSpan()
        return
    with current._tracer.start_span(
        name,
        input=input,
        attributes=attributes,
    ) as span:
        yield span


def get_current_span() -> Span | None:
    """Return the current recording span, if tracing is active."""

    return _CURRENT_SPAN.get()
