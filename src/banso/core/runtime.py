"""Minimal agent runtime loop."""

from time import perf_counter
from typing import Literal

from pydantic import BaseModel

from banso.core.action import AgentAction
from banso.core.executor import ActionExecutor
from banso.core.observation import Observation
from banso.core.policy import Policy
from banso.core.reducer import DefaultStateReducer, StateReducer
from banso.core.result import AgentResult
from banso.core.state import AgentState
from banso.tracing.trace import AgentTrace, TraceFailure, TraceStep


class RuntimeRunResult(BaseModel):
    """Runtime output with final result and execution trace."""

    result: AgentResult
    trace: AgentTrace


class RuntimeExecutionError(RuntimeError):
    """An unexpected runtime failure with its partial execution trace."""

    def __init__(self, *, trace: AgentTrace, original_error: Exception) -> None:
        super().__init__(f"{type(original_error).__name__}: {original_error}")
        self.trace = trace
        self.original_error = original_error


def _execution_error(
    *,
    trace: AgentTrace,
    error: Exception,
    phase: Literal["policy", "executor", "trace", "reducer"],
    step_index: int,
    state: AgentState,
    phase_duration_seconds: float,
    action: AgentAction | None = None,
    observation: Observation | None = None,
) -> RuntimeExecutionError:
    trace.status = "failed"
    trace.failure = TraceFailure(
        phase=phase,
        step_index=step_index,
        state=state.model_copy(deep=True),
        action=action,
        observation=observation,
        error_type=type(error).__name__,
        message=str(error),
        phase_duration_seconds=phase_duration_seconds,
    )
    return RuntimeExecutionError(trace=trace, original_error=error)


class AgentRuntime:
    """Coordinates policy decisions, action execution, and trace collection."""

    def __init__(
        self,
        policy: Policy,
        executor: ActionExecutor,
        reducer: StateReducer | None = None,
    ) -> None:
        self.policy = policy
        self.executor = executor
        self.reducer = reducer or DefaultStateReducer()

    async def run(self, state: AgentState) -> RuntimeRunResult:
        """Run the agent loop until it stops or reaches its step budget."""
        trace = AgentTrace(query=state.query)

        while not state.done and state.current_step < state.budget.max_steps:
            step_index = state.current_step
            state_before = state.model_copy(deep=True)

            # Select the next action with the policy.
            phase_started_at = perf_counter()
            try:
                action = await self.policy.select_action(state)
            except Exception as error:
                raise _execution_error(
                    trace=trace,
                    error=error,
                    phase="policy",
                    step_index=step_index,
                    state=state_before,
                    phase_duration_seconds=perf_counter() - phase_started_at,
                ) from error
            policy_duration_seconds = perf_counter() - phase_started_at

            # Execute the selected action.
            phase_started_at = perf_counter()
            try:
                observation = await self.executor.execute(action, state)
            except Exception as error:
                raise _execution_error(
                    trace=trace,
                    error=error,
                    phase="executor",
                    step_index=step_index,
                    state=state_before,
                    action=action,
                    phase_duration_seconds=perf_counter() - phase_started_at,
                ) from error
            executor_duration_seconds = perf_counter() - phase_started_at

            # Reduce the observation into the next state.
            phase_started_at = perf_counter()
            try:
                next_state = self.reducer.apply(state, action, observation)
            except Exception as error:
                raise _execution_error(
                    trace=trace,
                    error=error,
                    phase="reducer",
                    step_index=step_index,
                    state=state_before,
                    action=action,
                    observation=observation,
                    phase_duration_seconds=perf_counter() - phase_started_at,
                ) from error
            reducer_duration_seconds = perf_counter() - phase_started_at

            # Record the completed transition in the trace.
            phase_started_at = perf_counter()
            try:
                trace.steps.append(
                    TraceStep(
                        step_index=step_index,
                        state=state_before,
                        action=action,
                        observation=observation,
                        policy_duration_seconds=policy_duration_seconds,
                        executor_duration_seconds=executor_duration_seconds,
                        reducer_duration_seconds=reducer_duration_seconds,
                    )
                )
            except Exception as error:
                raise _execution_error(
                    trace=trace,
                    error=error,
                    phase="trace",
                    step_index=step_index,
                    state=state_before,
                    action=action,
                    observation=observation,
                    phase_duration_seconds=perf_counter() - phase_started_at,
                ) from error
            state = next_state

        result = AgentResult(state=state)
        trace.status = "completed"
        trace.final_result = result

        return RuntimeRunResult(result=result, trace=trace)
