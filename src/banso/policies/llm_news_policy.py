"""LLM-backed policy for selecting news agent actions."""

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from banso.core.action import AgentAction, AgentActionType
from banso.core.state import AgentState
from banso.llm import (
    LLMClient,
    LLMError,
    LLMMessage,
    LLMMessageRole,
    LLMRequest,
)
from banso.policies.news_policy_view import NewsPolicyStateViewBuilder


SYSTEM_PROMPT = (
    "You are the action-selection policy for a news research agent. Select exactly "
    "one next action from the supplied available_actions. Follow the action "
    "instructions and remaining budget. Return exactly one JSON object with exactly "
    'these top-level keys: "type", "params", and "rationale". Do not include '
    "markdown or additional explanation. The rationale must be a brief decision "
    "reason, not hidden chain-of-thought."
)

ACTION_INSTRUCTIONS = {
    AgentActionType.PLAN_SEARCH: "Create a search plan. Use an empty params object.",
    AgentActionType.SEARCH: (
        "Run one search. params must contain a non-empty query and may contain a "
        "non-empty intent."
    ),
    AgentActionType.READ_DOCUMENT: (
        "Read documents for collected search results. Use an empty params object."
    ),
    AgentActionType.EXTRACT_EVIDENCE: (
        "Extract evidence from collected documents. Use an empty params object."
    ),
    AgentActionType.SYNTHESIZE: (
        "Synthesize the answer from collected documents and evidence. Use an empty "
        "params object."
    ),
    AgentActionType.STOP: "Stop the agent. Use an empty params object.",
}


class LLMPolicyError(Exception):
    """Raised when an LLM policy cannot produce a valid action."""

    def __init__(
        self,
        message: str,
        *,
        reason: str,
        raw_output: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.raw_output = raw_output

    def __str__(self) -> str:
        message = super().__str__()
        if self.raw_output is None:
            return message
        return f"{message}; raw_output={self.raw_output!r}"


class _LLMActionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: AgentActionType
    params: dict[str, Any]
    rationale: str


class LLMNewsPolicy:
    """Selects bounded news agent actions with an LLM."""

    def __init__(
        self,
        client: LLMClient,
        view_builder: NewsPolicyStateViewBuilder,
        *,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.view_builder = view_builder
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def select_action(self, state: AgentState) -> AgentAction:
        """Build the policy view and return one validated action."""
        view = self.view_builder.build(state)
        search_count = self._search_count(state)
        remaining_searches = max(state.budget.max_searches - search_count, 0)
        available_actions = self._available_actions(state, remaining_searches)

        try:
            response = await self.client.generate(
                LLMRequest(
                    messages=[
                        LLMMessage(
                            role=LLMMessageRole.SYSTEM,
                            content=SYSTEM_PROMPT,
                        ),
                        LLMMessage(
                            role=LLMMessageRole.USER,
                            content=self._build_user_prompt(
                                view.model_dump(mode="json"),
                                state,
                                search_count,
                                remaining_searches,
                                available_actions,
                            ),
                        ),
                    ],
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            )
        except LLMError as error:
            raise LLMPolicyError(
                f"LLM action selection failed: {error}",
                reason="llm_error",
            ) from error

        try:
            output = self._parse_output(response.content)
            return self._validate_action(output, state, search_count)
        except LLMPolicyError as error:
            error.raw_output = response.content
            raise

    def _build_user_prompt(
        self,
        policy_state: dict[str, Any],
        state: AgentState,
        search_count: int,
        remaining_searches: int,
        available_actions: list[AgentActionType],
    ) -> str:
        payload = {
            "policy_state": policy_state,
            "remaining_budget": {
                "remaining_step_count": max(state.budget.max_steps - state.current_step, 0),
                "executed_search_count": search_count,
                "remaining_search_count": remaining_searches,
            },
            "available_actions": [action.value for action in available_actions],
            "action_instructions": {
                action.value: ACTION_INSTRUCTIONS[action]
                for action in available_actions
            },
            "output_schema": {
                "type": "one value from available_actions",
                "params": "an object following the selected action instruction",
                "rationale": "a brief non-empty decision reason",
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    def _parse_output(self, content: str) -> _LLMActionOutput:
        try:
            raw_output = json.loads(content)
        except json.JSONDecodeError as error:
            raise LLMPolicyError(
                "LLM policy response is not valid JSON",
                reason="invalid_json",
            ) from error

        try:
            return _LLMActionOutput.model_validate(raw_output)
        except ValidationError as error:
            raise LLMPolicyError(
                "LLM policy response has an invalid schema",
                reason="invalid_schema",
            ) from error

    def _validate_action(
        self,
        output: _LLMActionOutput,
        state: AgentState,
        search_count: int,
    ) -> AgentAction:
        rationale = output.rationale.strip()
        if not rationale:
            raise LLMPolicyError(
                "LLM policy response has an empty rationale",
                reason="invalid_params",
            )

        if output.type == AgentActionType.SEARCH:
            params = self._validate_search_params(output.params, state, search_count)
        else:
            if output.params:
                raise LLMPolicyError(
                    f"{output.type.value} action does not accept params",
                    reason="invalid_params",
                )
            params = {}

        self._validate_action_preconditions(output.type, state)
        return AgentAction(type=output.type, params=params, rationale=rationale)

    def _validate_search_params(
        self,
        params: dict[str, Any],
        state: AgentState,
        search_count: int,
    ) -> dict[str, str]:
        if set(params) - {"query", "intent"}:
            raise LLMPolicyError(
                "search action contains unsupported params",
                reason="invalid_params",
            )

        query_value = params.get("query")
        if not isinstance(query_value, str) or not (query := query_value.strip()):
            raise LLMPolicyError(
                "search action requires a non-empty query",
                reason="invalid_params",
            )

        normalized_params = {"query": query}
        if "intent" in params:
            intent_value = params["intent"]
            if not isinstance(intent_value, str) or not (
                intent := intent_value.strip()
            ):
                raise LLMPolicyError(
                    "search action intent must be a non-empty string",
                    reason="invalid_params",
                )
            normalized_params["intent"] = intent

        if search_count >= state.budget.max_searches:
            raise LLMPolicyError(
                "search action exceeds the search budget",
                reason="invalid_action",
            )

        normalized_query = query.casefold()
        for entry in state.action_history:
            previous_query = entry.params.get("query")
            if (
                entry.action_type == AgentActionType.SEARCH
                and isinstance(previous_query, str)
                and previous_query.strip().casefold() == normalized_query
            ):
                raise LLMPolicyError(
                    "search action repeats an executed query",
                    reason="invalid_action",
                )

        return normalized_params

    def _validate_action_preconditions(
        self,
        action_type: AgentActionType,
        state: AgentState,
    ) -> None:
        invalid_reason: str | None = None
        if (
            action_type == AgentActionType.PLAN_SEARCH
            and state.search_plan is not None
        ):
            invalid_reason = "a search plan already exists"
        elif (
            action_type == AgentActionType.READ_DOCUMENT
            and not state.search_result_ids
        ):
            invalid_reason = "no search results are available"
        elif (
            action_type == AgentActionType.EXTRACT_EVIDENCE
            and not state.document_ids
        ):
            invalid_reason = "no documents are available"
        elif (
            action_type == AgentActionType.SYNTHESIZE
            and not state.document_ids
            and not state.evidence_ids
        ):
            invalid_reason = "no documents or evidence are available"

        if invalid_reason is not None:
            raise LLMPolicyError(
                f"{action_type.value} action is not allowed: {invalid_reason}",
                reason="invalid_action",
            )

    @staticmethod
    def _search_count(state: AgentState) -> int:
        return sum(
            entry.action_type == AgentActionType.SEARCH
            for entry in state.action_history
        )

    @staticmethod
    def _available_actions(
        state: AgentState,
        remaining_searches: int,
    ) -> list[AgentActionType]:
        actions: list[AgentActionType] = []
        if state.search_plan is None:
            actions.append(AgentActionType.PLAN_SEARCH)
        if remaining_searches > 0:
            actions.append(AgentActionType.SEARCH)
        if state.search_result_ids:
            actions.append(AgentActionType.READ_DOCUMENT)
        if state.document_ids:
            actions.append(AgentActionType.EXTRACT_EVIDENCE)
        if state.document_ids or state.evidence_ids:
            actions.append(AgentActionType.SYNTHESIZE)
        actions.append(AgentActionType.STOP)
        return actions
