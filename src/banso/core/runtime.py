"""Minimal agent runtime loop and result models."""

from typing import Any

from pydantic import BaseModel, Field

from banso.core.executor import ActionExecutor
from banso.core.policy import Policy
from banso.core.reducer import DefaultStateReducer, StateReducer
from banso.core.state import AgentState
from banso.tracing import Tracer


class AgentResult(BaseModel):
    """Final result returned by an agent run."""

    state: AgentState
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeRunResult(BaseModel):
    """Runtime output with a correlation id for its execution trace."""

    result: AgentResult
    trace_id: str


class RuntimeExecutionError(RuntimeError):
    """An unexpected runtime failure correlated with its execution trace."""

    def __init__(self, *, trace_id: str, original_error: Exception) -> None:
        super().__init__(f"{type(original_error).__name__}: {original_error}")
        self.trace_id = trace_id
        self.original_error = original_error


class AgentRuntime:
    """Coordinates policy decisions, action execution, and trace collection."""

    def __init__(
        self,
        policy: Policy,
        executor: ActionExecutor,
        reducer: StateReducer | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.policy = policy
        self.executor = executor
        self.reducer = reducer or DefaultStateReducer()
        self.tracer = tracer or Tracer()

    async def run(self, state: AgentState) -> RuntimeRunResult:
        """Run the agent loop until it stops or reaches its step budget."""

        trace_id = ""
        try:
            with self.tracer.start_span(
                "agent.run",
                input={"query": state.query},
            ) as run_span:
                trace_id = run_span.trace_id

                while not state.done and state.current_step < state.budget.max_steps:
                    step_index = state.current_step
                    state_before = state.model_copy(deep=True)
                    with self.tracer.start_span(
                        "agent.step",
                        input={"state": state_before},
                        attributes={"step_index": step_index},
                    ) as step_span:
                        with self.tracer.start_span(
                            "agent.policy.select",
                            attributes={"step_index": step_index},
                        ):
                            action = await self.policy.select_action(state)
                        step_span.set_output({"action": action})

                        with self.tracer.start_span(
                            "agent.action.execute",
                            attributes={
                                "step_index": step_index,
                                "action_type": action.type.value,
                            },
                        ):
                            observation = await self.executor.execute(action, state)
                            if action.type != observation.type:
                                raise ValueError(
                                    f"{action.type.value} action returned "
                                    f"{observation.type.value} observation"
                                )
                        step_span.set_output(
                            {
                                "action": action,
                                "observation": observation,
                            }
                        )

                        with self.tracer.start_span(
                            "agent.state.reduce",
                            attributes={
                                "step_index": step_index,
                                "action_type": action.type.value,
                            },
                        ):
                            next_state = self.reducer.apply(state, action, observation)
                    state = next_state

                result = AgentResult(state=state)
                run_span.set_output({"result": result})
            return RuntimeRunResult(result=result, trace_id=trace_id)
        except Exception as error:
            raise RuntimeExecutionError(
                trace_id=trace_id,
                original_error=error,
            ) from error
