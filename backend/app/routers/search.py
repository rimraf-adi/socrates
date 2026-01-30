"""Simple search endpoint - single pass search + synthesis."""

import os
from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List
from langchain_groq import ChatGroq

from app.tools.searxng import search_searxng, format_results_for_llm, SearchResult


router = APIRouter()


class SearchResponse(BaseModel):
    """Response from simple search."""
    answer: str
    sources: List[dict]


@router.get("/search", response_model=SearchResponse)
async def simple_search(q: str = Query(..., description="Search query")):
    """
    Simple search - single pass search and synthesis.
    
    1. Search SearXNG
    2. Synthesize answer with Groq
    3. Return answer + sources
    """
    # Search
    results = await search_searxng(q, max_results=10)
    
    if not results:
        return SearchResponse(
            answer="No results found for your query.",
            sources=[],
        )
    
    # Format for LLM
    context = format_results_for_llm(results)
    
    # Synthesize with Groq
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
    
    prompt = f"""Based on the search results below, provide a clear, concise answer to the question.
Cite sources using [1], [2], etc. Keep the answer focused and under 300 words.
Use markdown formatting.

Question: {q}

Search Results:
{context}"""

    response = await llm.ainvoke(prompt)
    
    return SearchResponse(
        answer=response.content,
        sources=[r.model_dump() for r in results],
    )
