"""LangGraph research agent for deep research tasks.

This agent uses dynamic planning and adapts its research strategy based on the query type.
"""

import os
from typing import TypedDict, Annotated, List, Literal
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from app.tools.searxng import search_searxng, format_results_for_llm, SearchResult

LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"

# Global settings that can be changed at runtime
_current_model: str | None = None
_current_provider: str = "lmstudio"  # "lmstudio", "gemini", or "hybrid"

# Depth presets: maps depth name to max_iterations
DEPTH_PRESETS = {
    "quick": 5,
    "standard": 10,
    "deep": 15,
    "exhaustive": 20,
}


def set_current_model(model: str | None):
    """Set the current model to use for LLM calls."""
    global _current_model
    _current_model = model


def set_current_provider(provider: str):
    """Set the current provider (lmstudio, gemini, or hybrid)."""
    global _current_provider
    _current_provider = provider


def _get_lmstudio_llm(temperature: float = 0.4, model: str | None = None):
    """Get LMStudio LLM."""
    kwargs = {
        "base_url": LMSTUDIO_BASE_URL,
        "api_key": "lm-studio",
        "temperature": temperature,
    }
    if model:
        kwargs["model"] = model
    return ChatOpenAI(**kwargs)


def _get_gemini_llm(temperature: float = 0.4, model: str | None = None):
    """Get Gemini LLM."""
    gemini_model = model or "gemini-2.5-flash"
    return ChatGoogleGenerativeAI(
        model=gemini_model,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=temperature,
    )


def get_llm(temperature: float = 0.4, model: str | None = None, provider: str | None = None):
    """Get LLM from the specified provider with configurable temperature and model."""
    model_to_use = model or _current_model
    provider_to_use = provider or _current_provider
    
    if provider_to_use == "gemini":
        return _get_gemini_llm(temperature, model_to_use)
    else:
        # LMStudio (default, also used as fallback for hybrid light tasks)
        return _get_lmstudio_llm(temperature, model_to_use)


def get_llm_light(temperature: float = 0.4):
    """Get LLM for lightweight tasks (query analysis, sub-questions, evaluation).
    In hybrid mode: uses LMStudio. Otherwise: uses the current provider."""
    if _current_provider == "hybrid":
        return _get_lmstudio_llm(temperature, _current_model)
    return get_llm(temperature)


def get_llm_heavy(temperature: float = 0.4):
    """Get LLM for heavy tasks (synthesis, complex analysis).
    In hybrid mode: uses Gemini. Otherwise: uses the current provider."""
    if _current_provider == "hybrid":
        return _get_gemini_llm(temperature)
    return get_llm(temperature)


class ResearchState(TypedDict):
    """State for the research agent."""
    query: str
    query_type: str  # factual, comparative, exploratory, how-to, opinion
    research_depth: str  # quick, standard, deep
    sub_questions: List[str]
    current_question_idx: int
    search_results: List[dict]
    findings: List[str]
    subquery_answers: List[dict]  # Store individual subquery answers
    knowledge_gaps: List[str]
    final_answer: str
    iteration: int
    max_iterations: int
    status: str
    coverage_score: int  # 1-10 score for research completeness
    follow_up_queries: List[str]  # dynamically generated follow-up searches
    progress_messages: Annotated[list, add_messages]


QUERY_ANALYSIS_PROMPT = """You are an expert research strategist. Analyze this query and create a comprehensive research plan.

Query: "{query}"

Respond with a JSON object (and nothing else):
{{
    "query_type": "factual|comparative|exploratory|how-to|opinion|technical",
    "complexity": "simple|moderate|complex",
    "research_depth": "quick|standard|deep",
    "estimated_sources_needed": <number between 5-50>,
    "key_aspects": ["aspect1", "aspect2", ...],
    "sub_questions": ["search query 1", "search query 2", ...],
    "search_strategy": "explain your search approach"
}}

IMPORTANT - Generate SEARCH QUERIES, NOT questions!
Do NOT generate "wh" questions (what, why, when, how, where, who).
Generate SHORT KEYWORD-BASED SEARCH QUERIES like users type into search engines.

Examples of BAD queries:
- "What is the difference between React and Vue?"
- "How does machine learning work?"

Examples of GOOD queries:
- "React vs Vue performance comparison 2024"
- "machine learning fundamentals explained"
- "Next.js vs Remix benchmarks"
- "Python async await best practices"

GENERATE AS MANY QUERIES AS NEEDED FOR COMPREHENSIVE COVERAGE:
- Simple factual queries: 4-6 search queries
- Moderate complexity: 8-12 search queries  
- Complex/deep topics: 15-25 search queries covering ALL angles

For complex topics, ensure you cover:
1. Core concepts and fundamentals
2. Technical deep-dives and specifications
3. Comparisons and alternatives
4. Best practices and common pitfalls
5. Real-world examples and case studies
6. Recent developments and trends (2024/2025)
7. Expert opinions and community consensus
8. Edge cases and limitations

Query format: 2-6 keywords, include year for time-sensitive topics, use modifiers like "vs", "comparison", "tutorial", "best practices", "examples"""


SUBQUERY_ANSWER_PROMPT = """You are a research assistant answering a specific research subquery. Provide a comprehensive, self-contained answer to this subquery using ONLY the provided search results.

Subquery: {question}

Search Results:
{results}

Provide a COMPLETE ANSWER to this specific subquery:

1. ANSWER THE SUBQUERY DIRECTLY:
   - Start with a clear, direct answer
   - Include all relevant facts, statistics, and specifics
   - Use exact numbers, dates, and names from the sources
   - Quote or paraphrase key information

2. PROVIDE SUPPORTING DETAILS:
   - Explain the context and background
   - Include technical details where relevant
   - Note any important caveats or limitations

3. CITE YOUR SOURCES:
   - Use [1], [2], etc. to cite specific sources
   - Only include information that appears in the search results

Format your answer as:

## Answer to: {question}

[Your comprehensive answer - aim for 150-300 words that fully address the subquery]

### Key Facts
- [Fact 1] [source]
- [Fact 2] [source]
- [Fact 3] [source]

### Additional Context
[Any important nuances, caveats, or related information]

Be thorough but focused. This answer should stand alone and fully address the subquery."""


COVERAGE_EVALUATION_PROMPT = """You are a research quality analyst. Evaluate whether the research conducted so far provides comprehensive coverage of the topic.

Original Query: {query}

Research Completed So Far:
{findings_summary}

Number of sources analyzed: {source_count}
Research areas covered: {areas_covered}

Evaluate the research coverage and respond with a JSON object:
{{
    "coverage_score": <1-10, where 10 is completely comprehensive>,
    "should_continue": <true/false>,
    "missing_aspects": ["aspect1", "aspect2", ...],
    "follow_up_queries": ["search query 1", "search query 2", ...],
    "reasoning": "explain why more research is or isn't needed"
}}

Criteria for deciding to continue:
- Coverage score below 7 = definitely continue
- Important aspects of the topic not yet explored
- Contradictions that need resolution
- Only surface-level information found
- User likely expects more depth

Criteria for stopping:
- Coverage score 8+ with all key aspects addressed
- Diminishing returns (same info repeated)
- Already have 15+ quality sources
- All obvious angles have been explored

If should_continue is true, provide 3-8 follow_up_queries using the same keyword-based format (NOT questions)."""


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
    llm = get_llm_light(temperature=0.3)
    
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
    estimated_sources = analysis.get("estimated_sources_needed", 15)
    
    # Ensure we have good sub-questions
    if len(sub_questions) < 2:
        sub_questions = [state["query"]]
    
    # For deep mode, set high iteration limit - actual stopping is dynamic
    if research_depth == "deep":
        max_iterations = 50  # High limit, actual stopping decided by coverage evaluation
    else:
        max_iterations = {"quick": 6, "standard": 15}.get(research_depth, 15)
    
    return {
        **state,
        "query_type": query_type,
        "research_depth": research_depth,
        "sub_questions": sub_questions,
        "current_question_idx": 0,
        "max_iterations": max_iterations,
        "coverage_score": 0,
        "follow_up_queries": [],
        "status": "planned",
        "progress_messages": [
            f"📋 Query type: {query_type} | Depth: {research_depth}",
            f"🎯 Planning {len(sub_questions)} initial research areas (targeting ~{estimated_sources} sources)"
        ],
    }


async def search_web(state: ResearchState) -> ResearchState:
    """Search for the current sub-question with adaptive result count."""
    idx = state["current_question_idx"]
    if idx >= len(state["sub_questions"]):
        return state
    
    current_q = state["sub_questions"][idx]
    
    # More results for deep research
    max_results = {"quick": 4, "standard": 8, "deep": 12}.get(state.get("research_depth", "standard"), 8)
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
    """Answer each subquery within context and store the answer."""
    llm = get_llm_heavy(temperature=0.3)
    idx = state["current_question_idx"]
    current_q = state["sub_questions"][idx]
    
    # Get all results for this question
    results_per_question = {"quick": 4, "standard": 8, "deep": 12}.get(state.get("research_depth", "standard"), 8)
    recent_results = state["search_results"][-results_per_question:]
    formatted = format_results_for_llm([SearchResult(**r) for r in recent_results])
    
    # Generate a complete answer for this subquery
    prompt = SUBQUERY_ANSWER_PROMPT.format(
        question=current_q,
        results=formatted
    )
    
    response = await llm.ainvoke(prompt)
    
    # Store both findings and structured subquery answer
    new_findings = state["findings"].copy()
    new_findings.append(f"## Research Area {idx + 1}: {current_q}\n\n{response.content}")
    
    new_subquery_answers = state.get("subquery_answers", []).copy()
    new_subquery_answers.append({
        "subquery": current_q,
        "answer": response.content,
        "sources": recent_results
    })
    
    return {
        **state,
        "findings": new_findings,
        "subquery_answers": new_subquery_answers,
        "current_question_idx": idx + 1,
        "iteration": state["iteration"] + 1,
        "status": "analyzing",
        "progress_messages": [f"📊 Answered: {current_q[:50]}..."],
    }


async def synthesize_answer(state: ResearchState) -> ResearchState:
    """Synthesize final answer by summarizing all subquery answers."""
    llm = get_llm_heavy(temperature=0.5)
    
    # Build a structured summary of all subquery answers
    subquery_summaries = []
    for i, sq in enumerate(state.get("subquery_answers", [])):
        subquery_summaries.append(f"### Subquery {i+1}: {sq['subquery']}\n\n{sq['answer']}")
    
    all_subquery_answers = "\n\n---\n\n".join(subquery_summaries)
    
    # Create numbered source list
    sources = [SearchResult(**r) for r in state["search_results"]]
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
    
    # Modified synthesis prompt that summarizes subquery answers
    synthesis_prompt = f"""You are an expert researcher writing a comprehensive final report.

Original Question: {state["query"]}

You have already answered each subquery in detail. Here are all the subquery answers:

{all_subquery_answers}

Sources Used:
{source_list}

Your task is to synthesize ALL the subquery answers into ONE cohesive, comprehensive final answer.

IMPORTANT:
1. Include key information from EVERY subquery answer above - do not skip any
2. Organize the information logically, not just in the order of subqueries
3. Remove redundancy but preserve all unique facts and insights
4. Use specific facts, numbers, and citations from the subquery answers
5. Create a unified narrative that comprehensively answers the original question

{get_synthesis_prompt(state["query"], query_type, "", "").split("Research Findings:")[0]}

Write the final synthesized answer now:"""
    
    response = await llm.ainvoke(synthesis_prompt)
    
    return {
        **state,
        "final_answer": response.content,
        "status": "complete",
        "progress_messages": [f"✅ Research complete! Synthesized {len(state.get('subquery_answers', []))} subquery answers."],
    }


def should_continue_research(state: ResearchState) -> Literal["search", "evaluate", "synthesize"]:
    """Decide whether to continue researching, evaluate coverage, or synthesize."""
    research_depth = state.get("research_depth", "standard")
    
    # If there are more initial queries to process
    if state["current_question_idx"] < len(state["sub_questions"]):
        return "search"
    
    # If there are follow-up queries to process
    if state.get("follow_up_queries", []):
        return "search"
    
    # For deep mode, evaluate coverage dynamically
    if research_depth == "deep" and state["iteration"] < state["max_iterations"]:
        return "evaluate"
    
    # For quick/standard or if max iterations reached
    if state["iteration"] >= state["max_iterations"]:
        return "synthesize"
    
    return "synthesize"


async def evaluate_coverage(state: ResearchState) -> ResearchState:
    """Dynamically evaluate if more research is needed (for deep mode)."""
    import json
    llm = get_llm_light(temperature=0.3)
    
    # Summarize findings for evaluation
    findings_summary = "\n".join([
        f"- {finding[:200]}..." for finding in state["findings"][-10:]
    ])
    
    areas_covered = ", ".join(state["sub_questions"][:10])
    
    prompt = COVERAGE_EVALUATION_PROMPT.format(
        query=state["query"],
        findings_summary=findings_summary,
        source_count=len(state["search_results"]),
        areas_covered=areas_covered
    )
    
    response = await llm.ainvoke(prompt)
    content = response.content
    
    # Parse JSON response
    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            evaluation = json.loads(content[start:end])
        else:
            evaluation = {"coverage_score": 8, "should_continue": False}
    except:
        evaluation = {"coverage_score": 8, "should_continue": False}
    
    coverage_score = evaluation.get("coverage_score", 8)
    should_continue = evaluation.get("should_continue", False)
    follow_up_queries = evaluation.get("follow_up_queries", [])
    reasoning = evaluation.get("reasoning", "")
    
    # If continuing, add follow-up queries to sub_questions
    new_sub_questions = state["sub_questions"].copy()
    if should_continue and follow_up_queries:
        new_sub_questions.extend(follow_up_queries)
    
    progress_msg = f"🔬 Coverage: {coverage_score}/10"
    if should_continue:
        progress_msg += f" - Expanding research with {len(follow_up_queries)} more queries"
    else:
        progress_msg += " - Sufficient coverage achieved"
    
    return {
        **state,
        "coverage_score": coverage_score,
        "sub_questions": new_sub_questions,
        "follow_up_queries": [],  # Clear after adding to sub_questions
        "status": "evaluating",
        "progress_messages": [progress_msg],
    }


def should_continue_after_eval(state: ResearchState) -> Literal["search", "synthesize"]:
    """After evaluation, decide to continue or synthesize."""
    # If there are more queries to process
    if state["current_question_idx"] < len(state["sub_questions"]):
        return "search"
    # If coverage is good enough or max iterations reached
    return "synthesize"


def create_research_graph():
    """Create the LangGraph research workflow."""
    workflow = StateGraph(ResearchState)
    
    workflow.add_node("plan", analyze_and_plan)
    workflow.add_node("search", search_web)
    workflow.add_node("analyze", analyze_results)
    workflow.add_node("evaluate", evaluate_coverage)
    workflow.add_node("synthesize", synthesize_answer)
    
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "search")
    workflow.add_edge("search", "analyze")
    workflow.add_conditional_edges(
        "analyze",
        should_continue_research,
        {
            "search": "search",
            "evaluate": "evaluate",
            "synthesize": "synthesize",
        }
    )
    workflow.add_conditional_edges(
        "evaluate",
        should_continue_after_eval,
        {
            "search": "search",
            "synthesize": "synthesize",
        }
    )
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


research_graph = create_research_graph()


async def run_research(
    query: str, 
    on_progress=None, 
    model: str | None = None, 
    provider: str = "lmstudio",
    depth: str = "standard",
    max_iterations: int | None = None
):
    """Run the research agent with progress callbacks.
    
    Args:
        query: The research query
        on_progress: Callback for progress updates
        model: Optional model ID
        provider: Provider selection (lmstudio, gemini, or hybrid)
        depth: Research depth (quick, standard, deep, exhaustive)
        max_iterations: Override for max iterations (uses depth preset if not specified)
    """
    # Set the model and provider for this research session
    if model:
        set_current_model(model)
    set_current_provider(provider)
    
    # Determine max_iterations from depth preset or override
    iterations = max_iterations if max_iterations else DEPTH_PRESETS.get(depth, 10)
    
    initial_state: ResearchState = {
        "query": query,
        "query_type": "exploratory",
        "research_depth": depth,
        "sub_questions": [],
        "current_question_idx": 0,
        "search_results": [],
        "findings": [],
        "subquery_answers": [],
        "knowledge_gaps": [],
        "final_answer": "",
        "iteration": 0,
        "max_iterations": iterations,
        "status": "starting",
        "coverage_score": 0,
        "follow_up_queries": [],
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

