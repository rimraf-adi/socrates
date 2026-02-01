"""Deep research endpoint with SSE streaming."""

import json
import asyncio
from fastapi import APIRouter, Query
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from typing import List, AsyncGenerator, Optional

from app.agents.researcher import run_research
from app.tools.searxng import SearchResult
from app.services.storage import save_research


router = APIRouter()


class ResearchResponse(BaseModel):
    """Final response from deep research."""
    answer: str
    sources: List[dict]
    sub_questions: List[str]
    iterations: int


async def research_event_generator(
    query: str, 
    model: Optional[str] = None, 
    provider: str = "lmstudio",
    depth: str = "standard",
    max_iterations: Optional[int] = None
) -> AsyncGenerator[str, None]:
    """Generate SSE events for research progress."""
    
    progress_queue = asyncio.Queue()
    
    async def on_progress(event: dict):
        await progress_queue.put(event)
    
    # Start research in background
    async def run():
        try:
            result = await run_research(
                query, 
                on_progress, 
                model=model, 
                provider=provider,
                depth=depth,
                max_iterations=max_iterations
            )
            await progress_queue.put({"type": "complete", "data": result, "depth": depth, "provider": provider})
        except Exception as e:
            await progress_queue.put({"type": "error", "message": str(e)})
    
    task = asyncio.create_task(run())
    
    try:
        while True:
            try:
                event = await asyncio.wait_for(progress_queue.get(), timeout=60.0)
                
                if event.get("type") == "complete":
                    # Send final result
                    data = event["data"]
                    answer = data.get("final_answer", "")
                    sources = data.get("search_results", [])
                    sub_questions = data.get("sub_questions", [])
                    iterations = data.get("iteration", 0)
                    
                    # Save to ~/Documents/socrates
                    saved = None
                    try:
                        saved = save_research(
                            query=query,
                            answer=answer,
                            sources=sources,
                            sub_questions=sub_questions,
                            provider=event.get("provider", provider),
                            depth=event.get("depth", depth),
                            iterations=iterations,
                        )
                    except Exception as e:
                        print(f"Failed to save research: {e}")
                    
                    yield json.dumps({
                        "type": "complete",
                        "answer": answer,
                        "sources": sources,
                        "sub_questions": sub_questions,
                        "iterations": iterations,
                        "saved": saved,
                    })
                    break
                elif event.get("type") == "error":
                    yield json.dumps({
                        "type": "error",
                        "message": event.get("message", "Unknown error"),
                    })
                    break
                else:
                    # Progress update
                    yield json.dumps({
                        "type": "progress",
                        "node": event.get("node", ""),
                        "status": event.get("status", ""),
                        "message": event.get("message", ""),
                        "sub_questions": event.get("sub_questions", []),
                        "iteration": event.get("iteration", 0),
                    })
            except asyncio.TimeoutError:
                yield json.dumps({"type": "ping"})
    finally:
        task.cancel()



@router.get("/research")
async def deep_research(
    q: str = Query(..., description="Research query"),
    model: Optional[str] = Query(None, description="Model ID"),
    provider: str = Query("lmstudio", description="Provider: lmstudio, gemini, or hybrid"),
    depth: str = Query("standard", description="Depth: quick, standard, deep, exhaustive"),
    max_iterations: Optional[int] = Query(None, description="Override max iterations (5-20)")
):
    """
    Deep research with LangGraph agent.
    
    Returns Server-Sent Events (SSE) stream with:
    - Progress updates as the agent works
    - Final comprehensive answer
    """
    return EventSourceResponse(
        research_event_generator(q, model=model, provider=provider, depth=depth, max_iterations=max_iterations),
        media_type="text/event-stream",
    )


@router.get("/research/sync", response_model=ResearchResponse)
async def deep_research_sync(
    q: str = Query(..., description="Research query"),
    model: Optional[str] = Query(None, description="Model ID"),
    provider: str = Query("lmstudio", description="Provider: lmstudio, gemini, or hybrid"),
    depth: str = Query("standard", description="Depth: quick, standard, deep, exhaustive"),
    max_iterations: Optional[int] = Query(None, description="Override max iterations (5-20)")
):
    """
    Deep research without streaming (for testing).
    Waits for full result before returning.
    """
    result = await run_research(q, model=model, provider=provider, depth=depth, max_iterations=max_iterations)
    
    return ResearchResponse(
        answer=result.get("final_answer", ""),
        sources=result.get("search_results", []),
        sub_questions=result.get("sub_questions", []),
        iterations=result.get("iteration", 0),
    )


