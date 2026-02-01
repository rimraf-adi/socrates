"""Storage service for persisting research results to ~/Documents/socrates."""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import uuid


def get_storage_dir() -> Path:
    """Get the base storage directory for research results."""
    return Path.home() / "Documents" / "socrates"


def slugify(text: str, max_length: int = 50) -> str:
    """Convert text to a safe directory name."""
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^\w\s-]', '', text.lower())
    # Replace spaces with underscores
    text = re.sub(r'[\s]+', '_', text.strip())
    # Truncate to max length
    return text[:max_length]


def generate_research_id() -> str:
    """Generate a unique research ID."""
    return str(uuid.uuid4())[:8]


def save_research(
    query: str,
    answer: str,
    sources: List[dict],
    sub_questions: List[str],
    provider: str,
    depth: str,
    iterations: int,
) -> dict:
    """Save research results to ~/Documents/socrates.
    
    Returns:
        dict with research_id and path
    """
    base_dir = get_storage_dir()
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directory name: date_query-slug_id
    timestamp = datetime.now()
    date_str = timestamp.strftime("%Y-%m-%d")
    time_str = timestamp.strftime("%H-%M-%S")
    query_slug = slugify(query)
    research_id = generate_research_id()
    
    dir_name = f"{date_str}_{query_slug}_{research_id}"
    research_dir = base_dir / dir_name
    research_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        "id": research_id,
        "query": query,
        "provider": provider,
        "depth": depth,
        "iterations": iterations,
        "timestamp": timestamp.isoformat(),
        "source_count": len(sources),
        "sub_question_count": len(sub_questions),
    }
    
    with open(research_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save sources
    with open(research_dir / "sources.json", "w") as f:
        json.dump(sources, f, indent=2)
    
    # Generate markdown file with full research
    markdown_content = generate_research_markdown(
        query=query,
        answer=answer,
        sources=sources,
        sub_questions=sub_questions,
        metadata=metadata,
    )
    
    with open(research_dir / "research.md", "w") as f:
        f.write(markdown_content)
    
    return {
        "research_id": research_id,
        "path": str(research_dir),
        "dir_name": dir_name,
    }


def generate_research_markdown(
    query: str,
    answer: str,
    sources: List[dict],
    sub_questions: List[str],
    metadata: dict,
) -> str:
    """Generate a comprehensive markdown file for the research."""
    lines = [
        f"# {query}",
        "",
        f"*Research conducted on {metadata['timestamp'][:10]} at {metadata['timestamp'][11:16]}*",
        f"*Provider: {metadata['provider']} | Depth: {metadata['depth']} | Iterations: {metadata['iterations']}*",
        "",
        "---",
        "",
        "## Research Answer",
        "",
        answer,
        "",
    ]
    
    if sub_questions:
        lines.extend([
            "---",
            "",
            "## Sub-Questions Explored",
            "",
        ])
        for i, sq in enumerate(sub_questions, 1):
            lines.append(f"{i}. {sq}")
        lines.append("")
    
    if sources:
        lines.extend([
            "---",
            "",
            "## Sources",
            "",
        ])
        seen_urls = set()
        for i, source in enumerate(sources, 1):
            url = source.get("url", "")
            if url and url not in seen_urls:
                title = source.get("title", "Untitled")
                lines.append(f"{i}. [{title}]({url})")
                seen_urls.add(url)
        lines.append("")
    
    return "\n".join(lines)


def list_researches() -> List[dict]:
    """List all saved researches, sorted by date (newest first)."""
    base_dir = get_storage_dir()
    
    if not base_dir.exists():
        return []
    
    researches = []
    
    for dir_path in sorted(base_dir.iterdir(), reverse=True):
        if dir_path.is_dir():
            metadata_file = dir_path / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    metadata["dir_name"] = dir_path.name
                    metadata["path"] = str(dir_path)
                    researches.append(metadata)
                except:
                    continue
    
    return researches


def get_research(dir_name: str) -> Optional[dict]:
    """Get a specific research by directory name."""
    base_dir = get_storage_dir()
    research_dir = base_dir / dir_name
    
    if not research_dir.exists():
        return None
    
    result = {}
    
    # Load metadata
    metadata_file = research_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            result["metadata"] = json.load(f)
    
    # Load sources
    sources_file = research_dir / "sources.json"
    if sources_file.exists():
        with open(sources_file) as f:
            result["sources"] = json.load(f)
    
    # Load research content
    research_file = research_dir / "research.md"
    if research_file.exists():
        with open(research_file) as f:
            result["content"] = f.read()
    
    result["dir_name"] = dir_name
    result["path"] = str(research_dir)
    
    return result


def delete_research(dir_name: str) -> bool:
    """Delete a research by directory name."""
    import shutil
    
    base_dir = get_storage_dir()
    research_dir = base_dir / dir_name
    
    if research_dir.exists() and research_dir.is_dir():
        shutil.rmtree(research_dir)
        return True
    
    return False
