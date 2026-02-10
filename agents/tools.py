import httpx
from config import SEARXNG_BASE_URL


async def search_web(query: str, max_results: int = 5) -> str:
    params = {
        "q": query,
        "format": "json",
        "categories": "general",
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{SEARXNG_BASE_URL}/search",
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            results = []
            for i, item in enumerate(data.get("results", [])[:max_results], 1):
                title = item.get("title", "Untitled")
                url = item.get("url", "")
                snippet = item.get("content", "")
                results.append(f"[{i}] {title}\nURL: {url}\n{snippet}")

            return "\n\n".join(results) if results else "No search results found."

        except httpx.HTTPError:
            return "Search unavailable."
