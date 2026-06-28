"""Minimal agent runtime loop."""

from pydantic import BaseModel

from banso.core.executor import ActionExecutor
from banso.core.policy import Policy
from banso.core.reducer import DefaultStateReducer, StateReducer
from banso.core.result import AgentResult
from banso.core.state import AgentState
from banso.tracing.trace import AgentTrace, TraceStep


class RuntimeRunResult(BaseModel):
    """Runtime output with final result and execution trace."""

    result: AgentResult
    trace: AgentTrace


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
            action = await self.policy.select_action(state)
            observation = await self.executor.execute(action, state)

            trace.steps.append(
                TraceStep(
                    step_index=state.current_step,
                    state=state.model_copy(deep=True),
                    action=action,
                    observation=observation,
                )
            )

            state = self.reducer.apply(state, action, observation)

        result = AgentResult(state=state)
        trace.final_result = result

        return RuntimeRunResult(result=result, trace=trace)
