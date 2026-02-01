"""History endpoint for managing saved research."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.services.storage import list_researches, get_research, delete_research


router = APIRouter()


class ResearchSummary(BaseModel):
    """Summary of a saved research for listing."""
    id: str
    query: str
    provider: str
    depth: str
    timestamp: str
    source_count: int
    dir_name: str


class ResearchDetail(BaseModel):
    """Full research detail."""
    metadata: dict
    sources: List[dict]
    content: str
    dir_name: str
    path: str


@router.get("/history", response_model=List[ResearchSummary])
async def get_history():
    """Get list of all saved researches."""
    researches = list_researches()
    return [
        ResearchSummary(
            id=r.get("id", ""),
            query=r.get("query", ""),
            provider=r.get("provider", ""),
            depth=r.get("depth", ""),
            timestamp=r.get("timestamp", ""),
            source_count=r.get("source_count", 0),
            dir_name=r.get("dir_name", ""),
        )
        for r in researches
    ]


@router.get("/history/{dir_name:path}", response_model=ResearchDetail)
async def get_history_item(dir_name: str):
    """Get a specific saved research."""
    research = get_research(dir_name)
    
    if not research:
        raise HTTPException(status_code=404, detail="Research not found")
    
    return ResearchDetail(
        metadata=research.get("metadata", {}),
        sources=research.get("sources", []),
        content=research.get("content", ""),
        dir_name=research.get("dir_name", ""),
        path=research.get("path", ""),
    )


@router.delete("/history/{dir_name:path}")
async def delete_history_item(dir_name: str):
    """Delete a saved research."""
    success = delete_research(dir_name)
    
    if not success:
        raise HTTPException(status_code=404, detail="Research not found")
    
    return {"status": "deleted", "dir_name": dir_name}
