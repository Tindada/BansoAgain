"""LLM-backed search query planner implementation."""

import json

from pydantic import ValidationError

from banso.core.state import PlannedSearch, SearchPlan
from banso.llm import LLMClient, LLMMessage, LLMMessageRole, LLMRequest
from banso.retrieval.planner import SearchPlanningError, SearchPlanningRequest


SYSTEM_PROMPT = (
    "You are a news search planning assistant. Produce an ordered set of "
    "distinct, complementary search queries that together address the user's "
    "information need. Respect the supplied language, region, time range, and "
    "search limit. Return only valid JSON, with no markdown or explanation."
)

SEARCH_PLAN_OUTPUT_FORMAT = (
    "Return a JSON object in this schema:\n"
    "{\n"
    '  "searches": [\n'
    '    {"query": "...", "intent": "..."}\n'
    "  ]\n"
    "}"
)


class LLMSearchQueryPlanner:
    """Creates bounded search plans by calling an LLM client."""

    def __init__(
        self,
        client: LLMClient,
        model: str | None = None,
        temperature: float | None = 0.0,
        max_tokens: int | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def plan(self, request: SearchPlanningRequest) -> SearchPlan:
        if request.max_searches < 1:
            return SearchPlan()

        response = await self.client.generate(
            LLMRequest(
                messages=[
                    LLMMessage(
                        role=LLMMessageRole.SYSTEM,
                        content=SYSTEM_PROMPT,
                    ),
                    LLMMessage(
                        role=LLMMessageRole.USER,
                        content=self._build_user_prompt(request),
                    ),
                ],
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                metadata={"trace": {"operation": "search_query_planner.plan"}},
            )
        )
        return self._parse_plan(response.content, request.max_searches)

    def _build_user_prompt(self, request: SearchPlanningRequest) -> str:
        query = request.query
        return (
            f"User query:\n{query.text}\n\n"
            f"Language: {query.language or 'not specified'}\n"
            f"Region: {query.region or 'not specified'}\n"
            f"Time range: {query.time_range or 'not specified'}\n"
            f"Maximum searches: {request.max_searches}\n\n"
            "Make each query useful for a different information angle. "
            "Use a short intent describing that angle.\n\n"
            f"{SEARCH_PLAN_OUTPUT_FORMAT}"
        )

    def _parse_plan(self, content: str, max_searches: int) -> SearchPlan:
        try:
            raw_plan = json.loads(content)
        except json.JSONDecodeError as error:
            raise SearchPlanningError(
                "LLM search planning response is not valid JSON",
                reason="invalid_json",
            ) from error

        try:
            plan = SearchPlan.model_validate(raw_plan)
        except ValidationError as error:
            raise SearchPlanningError(
                "LLM search planning response has an invalid schema",
                reason="invalid_schema",
            ) from error

        searches: list[PlannedSearch] = []
        seen_queries: set[str] = set()
        for search in plan.searches:
            query = search.query.strip()
            intent = search.intent.strip()
            if not query or not intent:
                raise SearchPlanningError(
                    "LLM search planning response contains an invalid search",
                    reason="invalid_schema",
                )

            normalized_query = query.casefold()
            if normalized_query in seen_queries:
                continue
            seen_queries.add(normalized_query)
            searches.append(PlannedSearch(query=query, intent=intent))
            if len(searches) == max_searches:
                break

        if not searches:
            raise SearchPlanningError(
                "LLM search planning response contains no usable searches",
                reason="empty_plan",
            )

        return SearchPlan(searches=searches)
