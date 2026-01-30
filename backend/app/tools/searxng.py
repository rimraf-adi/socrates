"""SearXNG search tool for LangGraph agents."""

import os
import httpx
from typing import List
from pydantic import BaseModel, Field


SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "http://localhost:8080")


class SearchResult(BaseModel):
    """A single search result."""
    title: str
    url: str
    snippet: str


class SearchInput(BaseModel):
    """Input for the search tool."""
    query: str = Field(description="The search query to look up")


async def search_searxng(query: str, max_results: int = 5) -> List[SearchResult]:
    """
    Search using SearXNG and return results.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return
        
    Returns:
        List of SearchResult objects
    """
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
            for item in data.get("results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", "Untitled"),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                ))
            
            return results
            
        except httpx.HTTPError as e:
            print(f"SearXNG error: {e}")
            return []


def format_results_for_llm(results: List[SearchResult]) -> str:
    """Format search results as a string for the LLM."""
    if not results:
        return "No search results found."
    
    formatted = []
    for i, r in enumerate(results, 1):
        formatted.append(f"[{i}] {r.title}\nURL: {r.url}\n{r.snippet}")
    
    return "\n\n".join(formatted)
