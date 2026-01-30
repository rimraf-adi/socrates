"""LangGraph research agent for deep research tasks.

This agent uses dynamic planning and adapts its research strategy based on the query type.
"""

import os
from typing import TypedDict, Annotated, List, Literal
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from app.tools.searxng import search_searxng, format_results_for_llm, SearchResult


def get_llm(temperature: float = 0.4):
    """Get LLM with configurable temperature for varied outputs."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
    )


class ResearchState(TypedDict):
    """State for the research agent."""
    query: str
    query_type: str  # factual, comparative, exploratory, how-to, opinion
    research_depth: str  # quick, standard, deep
    sub_questions: List[str]
    current_question_idx: int
    search_results: List[dict]
    findings: List[str]
    knowledge_gaps: List[str]
    final_answer: str
    iteration: int
    max_iterations: int
    status: str
    progress_messages: Annotated[list, add_messages]


QUERY_ANALYSIS_PROMPT = """You are an expert research strategist. Analyze this query and determine the optimal research approach.

Query: "{query}"

Respond with a JSON object (and nothing else):
{{
    "query_type": "factual|comparative|exploratory|how-to|opinion|technical",
    "complexity": "simple|moderate|complex",
    "research_depth": "quick|standard|deep",
    "key_aspects": ["aspect1", "aspect2", ...],
    "sub_questions": ["question1", "question2", ...],
    "search_strategy": "explain your search approach in one sentence"
}}

Guidelines:
- For FACTUAL queries (dates, definitions, simple facts): 2-3 focused sub-questions
- For COMPARATIVE queries (X vs Y, pros/cons): 4-5 sub-questions covering each option
- For EXPLORATORY queries (how, why, complex topics): 5-7 sub-questions from multiple angles  
- For HOW-TO queries: step-by-step breakdown
- For OPINION/DEBATE topics: include multiple perspectives

Generate sub-questions that are:
1. Specific and searchable
2. Cover different angles (technical, practical, historical, future)
3. Build on each other logically"""


ANALYSIS_PROMPT_TEMPLATE = """You are a meticulous research analyst. Extract comprehensive insights from these search results.

Research Question: {question}

Search Results:
{results}

Instructions:
1. Extract ALL relevant facts, statistics, dates, and quotes
2. Note any contradictions or different perspectives
3. Identify what's still unclear or needs more research
4. Be thorough - this is for a comprehensive research report

Format your response as:

### Key Findings
- [Detailed finding with specific data] [1]
- [Another finding with context] [2]
...

### Important Details
- [Specific numbers, dates, names mentioned]
- [Notable quotes or claims]

### Gaps & Uncertainties
- [What information is missing or unclear]
- [Conflicting information found]

Be detailed and specific. Do NOT summarize too briefly - we need the full picture."""


def get_synthesis_prompt(query: str, query_type: str, findings: str, sources: str) -> str:
    """Generate dynamic synthesis prompt based on query type."""
    
    base_instruction = f"""You are an expert researcher writing a comprehensive report. 
Your task is to synthesize all research findings into a detailed, well-structured answer.

Original Question: {query}

Research Findings:
{findings}

Sources:
{sources}

"""
    
    if query_type == "comparative":
        format_instruction = """
Structure your response as:

# Comparative Analysis

## Overview
[2-3 sentence summary of what's being compared]

## Detailed Comparison

### [Option A]
**Strengths:**
- [Point with evidence] [citation]
- [Point with evidence] [citation]

**Weaknesses:**
- [Point with evidence] [citation]

### [Option B]
**Strengths:**
- [Point with evidence] [citation]

**Weaknesses:**
- [Point with evidence] [citation]

## Head-to-Head Analysis
| Aspect | Option A | Option B |
|--------|----------|----------|
| ... | ... | ... |

## Recommendation
[Nuanced recommendation based on use cases]

## Key Takeaways
- [Main insight 1]
- [Main insight 2]
- [Main insight 3]
"""
    
    elif query_type == "how-to":
        format_instruction = """
Structure your response as:

# How To: [Topic]

## Quick Summary
[1-2 sentences on what this achieves]

## Prerequisites
- [What you need before starting]

## Step-by-Step Guide

### Step 1: [Action]
[Detailed explanation with specifics]
- Key consideration: [important detail]

### Step 2: [Action]
[Detailed explanation]

[Continue for all steps...]

## Common Mistakes to Avoid
- ❌ [Mistake 1] - [Why it's a problem]
- ❌ [Mistake 2] - [Why it's a problem]

## Pro Tips
- 💡 [Advanced tip 1]
- 💡 [Advanced tip 2]

## Troubleshooting
| Problem | Solution |
|---------|----------|
| ... | ... |
"""

    elif query_type == "technical":
        format_instruction = """
Structure your response as:

# Technical Analysis: [Topic]

## Executive Summary
[Technical overview in 2-3 sentences]

## Technical Deep Dive

### Architecture/Mechanism
[How it works technically]

### Key Components
1. **[Component 1]**: [Explanation]
2. **[Component 2]**: [Explanation]

### Technical Specifications
| Specification | Value/Detail |
|--------------|--------------|
| ... | ... |

## Implementation Considerations
- [Technical consideration 1]
- [Technical consideration 2]

## Performance & Limitations
**Strengths:**
- [Technical strength]

**Limitations:**
- [Technical limitation]

## Conclusion
[Technical summary and recommendations]
"""

    elif query_type == "exploratory":
        format_instruction = """
Structure your response as:

# Deep Dive: [Topic]

## The Big Picture
[Comprehensive overview - 3-4 sentences setting context]

## Background & Context
[Historical context, why this matters, how we got here]

## Core Concepts

### [Concept 1]
[Detailed explanation with examples and evidence]

### [Concept 2]  
[Detailed explanation with examples and evidence]

## Current State of Affairs
[What's happening now, recent developments]

## Different Perspectives
- **[Perspective 1]**: [Explanation with supporting evidence]
- **[Perspective 2]**: [Explanation with supporting evidence]

## Implications & Future Outlook
[What this means, where things are heading]

## Key Takeaways
1. [Major insight with supporting detail]
2. [Major insight with supporting detail]
3. [Major insight with supporting detail]

## Further Reading
[Areas to explore for more depth]
"""

    else:  # factual or default
        format_instruction = """
Structure your response as:

# [Topic]

## Summary
[Clear, direct answer in 2-3 sentences]

## Detailed Explanation

### [Key Aspect 1]
[Thorough explanation with specific facts, dates, numbers]
- [Supporting detail] [citation]
- [Supporting detail] [citation]

### [Key Aspect 2]
[Thorough explanation with context]
- [Supporting detail] [citation]

## Important Context
[Background information that helps understand this better]

## Key Facts at a Glance
| Aspect | Detail |
|--------|--------|
| ... | ... |

## Bottom Line
[Final synthesis - what the user should take away]
"""

    return base_instruction + format_instruction + """

IMPORTANT GUIDELINES:
- Be comprehensive and detailed - this is a research report, not a quick summary
- Use specific facts, numbers, dates, and quotes from the research
- Cite sources using [1], [2], etc. matching the source numbers provided
- Include relevant context and nuance
- Acknowledge uncertainties or conflicting information
- Write naturally and vary your sentence structure
- Aim for depth over brevity - the user expects thorough research
"""


async def analyze_and_plan(state: ResearchState) -> ResearchState:
    """Analyze query and create dynamic research plan."""
    llm = get_llm(temperature=0.3)
    
    prompt = QUERY_ANALYSIS_PROMPT.format(query=state["query"])
    response = await llm.ainvoke(prompt)
    content = response.content
    
    # Parse JSON response
    import json
    try:
        # Extract JSON from response
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            analysis = json.loads(content[start:end])
        else:
            analysis = {}
    except:
        analysis = {}
    
    query_type = analysis.get("query_type", "exploratory")
    sub_questions = analysis.get("sub_questions", [state["query"]])
    research_depth = analysis.get("research_depth", "standard")
    
    # Ensure we have good sub-questions
    if len(sub_questions) < 2:
        sub_questions = [state["query"]]
    
    max_iterations = {"quick": 6, "standard": 10, "deep": 15}.get(research_depth, 10)
    
    return {
        **state,
        "query_type": query_type,
        "research_depth": research_depth,
        "sub_questions": sub_questions,
        "current_question_idx": 0,
        "max_iterations": max_iterations,
        "status": "planned",
        "progress_messages": [
            f"📋 Query type: {query_type} | Depth: {research_depth}",
            f"🎯 Planning {len(sub_questions)} research areas"
        ],
    }


async def search_web(state: ResearchState) -> ResearchState:
    """Search for the current sub-question with adaptive result count."""
    idx = state["current_question_idx"]
    if idx >= len(state["sub_questions"]):
        return state
    
    current_q = state["sub_questions"][idx]
    
    # More results for deep research
    max_results = {"quick": 4, "standard": 6, "deep": 8}.get(state.get("research_depth", "standard"), 6)
    results = await search_searxng(current_q, max_results=max_results)
    
    new_results = state["search_results"].copy()
    new_results.extend([r.model_dump() for r in results])
    
    return {
        **state,
        "search_results": new_results,
        "status": "searching",
        "progress_messages": [f"🔍 Searching: {current_q[:60]}..."],
    }


async def analyze_results(state: ResearchState) -> ResearchState:
    """Extract detailed findings from search results."""
    llm = get_llm(temperature=0.3)
    idx = state["current_question_idx"]
    current_q = state["sub_questions"][idx]
    
    # Get all results for this question
    results_per_question = {"quick": 4, "standard": 6, "deep": 8}.get(state.get("research_depth", "standard"), 6)
    recent_results = state["search_results"][-results_per_question:]
    formatted = format_results_for_llm([SearchResult(**r) for r in recent_results])
    
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        question=current_q,
        results=formatted
    )
    
    response = await llm.ainvoke(prompt)
    
    new_findings = state["findings"].copy()
    new_findings.append(f"## Research Area {idx + 1}: {current_q}\n\n{response.content}")
    
    # Extract knowledge gaps for potential follow-up
    new_gaps = state.get("knowledge_gaps", []).copy()
    if "Gaps" in response.content or "unclear" in response.content.lower():
        new_gaps.append(f"Gap from Q{idx+1}: Review needed")
    
    return {
        **state,
        "findings": new_findings,
        "knowledge_gaps": new_gaps,
        "current_question_idx": idx + 1,
        "iteration": state["iteration"] + 1,
        "status": "analyzing",
        "progress_messages": [f"📊 Analyzed: {current_q[:50]}..."],
    }


async def synthesize_answer(state: ResearchState) -> ResearchState:
    """Synthesize comprehensive, dynamic answer based on query type."""
    llm = get_llm(temperature=0.5)  # Slightly higher for more natural writing
    
    all_findings = "\n\n---\n\n".join(state["findings"])
    sources = [SearchResult(**r) for r in state["search_results"]]
    
    # Create numbered source list
    unique_sources = []
    seen_urls = set()
    for s in sources:
        if s.url not in seen_urls:
            unique_sources.append(s)
            seen_urls.add(s.url)
    
    source_list = "\n".join([
        f"[{i+1}] {s.title}\n    URL: {s.url}" 
        for i, s in enumerate(unique_sources[:25])
    ])
    
    query_type = state.get("query_type", "exploratory")
    prompt = get_synthesis_prompt(state["query"], query_type, all_findings, source_list)
    
    response = await llm.ainvoke(prompt)
    
    return {
        **state,
        "final_answer": response.content,
        "status": "complete",
        "progress_messages": [f"✅ Research complete! Compiled {len(state['findings'])} research areas."],
    }


def should_continue_research(state: ResearchState) -> Literal["search", "synthesize"]:
    """Decide whether to continue researching or synthesize."""
    if state["current_question_idx"] >= len(state["sub_questions"]):
        return "synthesize"
    if state["iteration"] >= state["max_iterations"]:
        return "synthesize"
    return "search"


def create_research_graph():
    """Create the LangGraph research workflow."""
    workflow = StateGraph(ResearchState)
    
    workflow.add_node("plan", analyze_and_plan)
    workflow.add_node("search", search_web)
    workflow.add_node("analyze", analyze_results)
    workflow.add_node("synthesize", synthesize_answer)
    
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "analyze")
    workflow.add_conditional_edges(
        "analyze",
        should_continue_research,
        {
            "search": "search",
            "synthesize": "synthesize",
        }
    )
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


research_graph = create_research_graph()


async def run_research(query: str, on_progress=None):
    """Run the research agent with progress callbacks."""
    initial_state: ResearchState = {
        "query": query,
        "query_type": "exploratory",
        "research_depth": "standard",
        "sub_questions": [],
        "current_question_idx": 0,
        "search_results": [],
        "findings": [],
        "knowledge_gaps": [],
        "final_answer": "",
        "iteration": 0,
        "max_iterations": 10,
        "status": "starting",
        "progress_messages": [],
    }
    
    async for event in research_graph.astream(initial_state):
        for node_name, state in event.items():
            if on_progress and state.get("progress_messages"):
                for msg in state["progress_messages"]:
                    await on_progress({
                        "node": node_name,
                        "status": state.get("status", ""),
                        "message": msg,
                        "sub_questions": state.get("sub_questions", []),
                        "iteration": state.get("iteration", 0),
                        "query_type": state.get("query_type", ""),
                    })
    
    final_state = await research_graph.ainvoke(initial_state)
    return final_state
