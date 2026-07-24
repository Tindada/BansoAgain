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
from banso.policies.news_policy_context import (
    NewsPolicyContext,
    NewsPolicyContextBuilder,
    SearchAttempt,
)


SYSTEM_PROMPT = (
    "You are the action-selection policy for a news research agent. Select exactly "
    "one next action from the supplied available_actions. Follow the action "
    "instructions and remaining budget. Return exactly one JSON object with exactly "
    'these top-level keys: "type", "params", and "rationale". Do not include '
    "markdown or additional explanation. The rationale must be a brief decision "
    "reason, not hidden chain-of-thought."
)

ACTION_INSTRUCTIONS = {
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
    AgentActionType.FINISH: (
        "Synthesize the final answer from collected documents and evidence, then "
        "finish the agent. Use an empty params object."
    ),
    AgentActionType.STOP: (
        "Stop without generating a new final answer. Use an empty params object."
    ),
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
        context_builder: NewsPolicyContextBuilder,
        *,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.context_builder = context_builder
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def select_action(self, state: AgentState) -> AgentAction:
        """Build the decision context and return one validated action."""
        context = self.context_builder.build(state)
        available_actions = self._available_actions(context)

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
                                context.model_dump(mode="json", exclude_none=True),
                                available_actions,
                            ),
                        ),
                    ],
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    metadata={"trace": {"operation": "news_policy.select_action"}},
                )
            )
        except LLMError as error:
            raise LLMPolicyError(
                f"LLM action selection failed: {error}",
                reason="llm_error",
            ) from error

        try:
            output = self._parse_output(response.content)
            return self._validate_action(
                output,
                context,
                available_actions,
            )
        except LLMPolicyError as error:
            error.raw_output = response.content
            raise

    def _build_user_prompt(
        self,
        context: dict[str, Any],
        available_actions: list[AgentActionType],
    ) -> str:
        payload = {
            "context": context,
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
        context: NewsPolicyContext,
        available_actions: list[AgentActionType],
    ) -> AgentAction:
        rationale = output.rationale.strip()
        if not rationale:
            raise LLMPolicyError(
                "LLM policy response has an empty rationale",
                reason="invalid_params",
            )

        if output.type not in available_actions:
            raise LLMPolicyError(
                f"{output.type.value} action is not currently available",
                reason="invalid_action",
            )

        if output.type == AgentActionType.SEARCH:
            params = self._validate_search_params(output.params, context)
        else:
            if output.params:
                raise LLMPolicyError(
                    f"{output.type.value} action does not accept params",
                    reason="invalid_params",
                )
            params = {}

        return AgentAction(type=output.type, params=params, rationale=rationale)

    def _validate_search_params(
        self,
        params: dict[str, Any],
        context: NewsPolicyContext,
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

        if context.remaining_search_count == 0:
            raise LLMPolicyError(
                "search action exceeds the search budget",
                reason="invalid_action",
            )

        normalized_query = query.casefold()
        for attempt in context.attempts:
            if (
                isinstance(attempt, SearchAttempt)
                and attempt.query.strip().casefold() == normalized_query
            ):
                raise LLMPolicyError(
                    "search action repeats an executed query",
                    reason="invalid_action",
                )

        return normalized_params

    @staticmethod
    def _available_actions(
        context: NewsPolicyContext,
    ) -> list[AgentActionType]:
        actions: list[AgentActionType] = []
        if context.remaining_search_count > 0:
            actions.append(AgentActionType.SEARCH)
        if context.search_result_count > 0:
            actions.append(AgentActionType.READ_DOCUMENT)
        if context.document_count > 0:
            actions.append(AgentActionType.EXTRACT_EVIDENCE)
        if context.document_count > 0 or context.evidence_count > 0:
            actions.append(AgentActionType.FINISH)
        actions.append(AgentActionType.STOP)
        return actions
