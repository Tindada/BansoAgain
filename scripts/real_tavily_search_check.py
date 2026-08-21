"""Manual smoke check for the Tavily retrieval provider.

Run with:
UV_CACHE_DIR=.uv-cache uv run python scripts/real_tavily_search_check.py
"""

import asyncio
import os

from dotenv import load_dotenv

from banso.retrieval.provider import SearchRequest
from banso.retrieval.tavily_provider import TavilyRetrievalProvider


def build_tavily_provider() -> TavilyRetrievalProvider:
    api_key = os.getenv("BANSO_TAVILY_API_KEY")
    base_url = os.getenv("BANSO_TAVILY_BASE_URL", "https://api.tavily.com")

    if not api_key:
        raise RuntimeError("BANSO_TAVILY_API_KEY is required in .env")

    return TavilyRetrievalProvider(
        api_key=api_key,
        base_url=base_url,
    )


async def main() -> None:
    load_dotenv()

    query = os.getenv("BANSO_NEWS_QUERY", "latest AI news")
    max_results = int(os.getenv("BANSO_TAVILY_MAX_RESULTS", "5"))
    time_range = os.getenv("BANSO_TAVILY_TIME_RANGE")

    provider = build_tavily_provider()
    results = await provider.search(
        SearchRequest(
            query=query,
            max_results=max_results,
            time_range=time_range,
        )
    )

    print("query:", query)
    print("results:", len(results))
    for index, result in enumerate(results, start=1):
        print(f"{index}. title: {result.title}")
        print(f"   url: {result.url}")
        print(f"   rank: {result.rank}")
        print(f"   score: {result.metadata.get('score')}")
        print(f"   snippet: {result.snippet}")


if __name__ == "__main__":
    asyncio.run(main())
