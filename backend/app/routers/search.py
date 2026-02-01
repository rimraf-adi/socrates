"""Simple search endpoint - single pass search + synthesis with adaptive prompts."""

import os
from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from app.tools.searxng import search_searxng, format_results_for_llm, SearchResult

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"


router = APIRouter()


class SearchResponse(BaseModel):
    """Response from simple search."""
    answer: str
    sources: List[dict]
    query_type: str


def detect_query_type(query: str) -> str:
    """Quick heuristic to detect query type for appropriate response style."""
    q_lower = query.lower()
    
    if any(word in q_lower for word in ["vs", "versus", "compare", "difference between", "better"]):
        return "comparative"
    elif any(word in q_lower for word in ["how to", "how do i", "steps to", "guide", "tutorial"]):
        return "how-to"
    elif any(word in q_lower for word in ["what is", "define", "meaning of", "who is", "when was"]):
        return "factual"
    elif any(word in q_lower for word in ["why", "explain", "how does", "what causes"]):
        return "explanatory"
    elif any(word in q_lower for word in ["best", "top", "recommend", "should i"]):
        return "recommendation"
    else:
        return "general"


def get_simple_search_prompt(query: str, query_type: str, context: str) -> str:
    """Generate adaptive prompt based on query type."""
    
    base = f"""You are Socrates, an insightful AI research assistant. Based on the search results, provide a helpful answer.

Question: {query}

Search Results:
{context}

"""
    
    if query_type == "comparative":
        instruction = """Provide a balanced comparison:

## Quick Answer
[Direct 1-2 sentence verdict]

## Comparison

**[Option A]:**
- Pros: [key advantages]
- Cons: [key disadvantages]
- Best for: [use case]

**[Option B]:**
- Pros: [key advantages]  
- Cons: [key disadvantages]
- Best for: [use case]

## Bottom Line
[Recommendation based on different scenarios]

Cite sources with [1], [2], etc."""

    elif query_type == "how-to":
        instruction = """Provide clear, actionable steps:

## Quick Answer
[What this achieves in one sentence]

## Steps

1. **[First Step]**
   [Clear instruction with specifics]

2. **[Second Step]**
   [Clear instruction with specifics]

[Continue as needed...]

## Tips
- [Helpful tip]
- [Common mistake to avoid]

Cite sources with [1], [2], etc."""

    elif query_type == "factual":
        instruction = """Provide a direct, accurate answer:

## Answer
[Clear, direct response to the question]

## Key Details
- [Important fact or detail] [1]
- [Additional relevant information] [2]
- [Context that helps understanding]

## Additional Context
[Any helpful background or nuance]

Cite sources with [1], [2], etc."""

    elif query_type == "explanatory":
        instruction = """Provide a clear explanation:

## In Brief
[Simple explanation in 2-3 sentences]

## How It Works
[More detailed explanation with specifics]

## Key Points
- [Important concept 1] [1]
- [Important concept 2] [2]
- [Important concept 3]

## Why It Matters
[Practical implications or context]

Cite sources with [1], [2], etc."""

    elif query_type == "recommendation":
        instruction = """Provide helpful recommendations:

## Top Recommendation
[Your main suggestion with reasoning]

## Other Options to Consider
1. **[Option 1]** - [Why it's good] [1]
2. **[Option 2]** - [Why it's good] [2]
3. **[Option 3]** - [Why it's good]

## Things to Consider
- [Factor 1 to keep in mind]
- [Factor 2 to keep in mind]

## My Take
[Personal synthesis/recommendation]

Cite sources with [1], [2], etc."""

    else:  # general
        instruction = """Provide a helpful, informative answer:

## Answer
[Clear response addressing the question]

## Key Points
- [Main point 1] [1]
- [Main point 2] [2]
- [Main point 3]

## Additional Information
[Relevant context or details that add value]

Cite sources with [1], [2], etc."""

    return base + instruction + """

GUIDELINES:
- Be direct and get to the point quickly
- Use specific facts from the search results
- Keep the tone conversational but informative
- Vary your sentence structure naturally
- Don't just list information - synthesize and explain
"""


def get_llm_for_search(model: Optional[str], provider: str, temperature: float = 0.4):
    """Get the appropriate LLM based on provider."""
    if provider == "gemini":
        gemini_model = model or "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(
            model=gemini_model,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )
    else:
        # LMStudio (default)
        llm_kwargs = {"base_url": LMSTUDIO_BASE_URL, "api_key": "lm-studio", "temperature": temperature}
        if model:
            llm_kwargs["model"] = model
        return ChatOpenAI(**llm_kwargs)


@router.get("/search", response_model=SearchResponse)
async def simple_search(
    q: str = Query(..., description="Search query"),
    model: Optional[str] = Query(None, description="Model ID"),
    provider: str = Query("lmstudio", description="Provider: lmstudio or gemini")
):
    """
    Simple search - single pass search and synthesis with adaptive response format.
    """
    # Detect query type
    query_type = detect_query_type(q)
    
    # Search
    results = await search_searxng(q, max_results=10)
    
    if not results:
        return SearchResponse(
            answer="I couldn't find any results for your query. Try rephrasing or being more specific.",
            sources=[],
            query_type=query_type,
        )
    
    # Format for LLM
    context = format_results_for_llm(results)
    
    # Generate adaptive prompt
    prompt = get_simple_search_prompt(q, query_type, context)
    
    # Get LLM based on provider
    llm = get_llm_for_search(model, provider)
    response = await llm.ainvoke(prompt)
    
    return SearchResponse(
        answer=response.content,
        sources=[r.model_dump() for r in results],
        query_type=query_type,
    )

