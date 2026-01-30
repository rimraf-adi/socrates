"""LangGraph research agent for deep research tasks."""

import os
from typing import TypedDict, Annotated, List, Literal
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from app.tools.searxng import search_searxng, format_results_for_llm, SearchResult


# Initialize Groq LLM
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
    )


class ResearchState(TypedDict):
    """State for the research agent."""
    query: str
    sub_questions: List[str]
    current_question_idx: int
    search_results: List[dict]
    findings: List[str]
    final_answer: str
    iteration: int
    max_iterations: int
    status: str
    progress_messages: Annotated[list, add_messages]


async def plan_research(state: ResearchState) -> ResearchState:
    """Break down the query into sub-questions."""
    llm = get_llm()
    
    prompt = f"""Break down this research question into 3-5 specific sub-questions that together will provide a comprehensive answer.

Question: {state["query"]}

Output ONLY a Python list of strings, nothing else. Example:
["What is X?", "How does X work?", "What are the benefits?"]"""

    response = await llm.ainvoke(prompt)
    content = response.content
    
    # Parse the list from response
    try:
        import ast
        # Find list in response
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end > start:
            sub_questions = ast.literal_eval(content[start:end])
        else:
            sub_questions = [state["query"]]
    except:
        sub_questions = [state["query"]]
    
    return {
        **state,
        "sub_questions": sub_questions,
        "current_question_idx": 0,
        "status": "planned",
        "progress_messages": [f"📋 Identified {len(sub_questions)} research areas"],
    }


async def search_web(state: ResearchState) -> ResearchState:
    """Search for the current sub-question."""
    idx = state["current_question_idx"]
    if idx >= len(state["sub_questions"]):
        return state
    
    current_q = state["sub_questions"][idx]
    results = await search_searxng(current_q, max_results=5)
    
    # Store results
    new_results = state["search_results"].copy()
    new_results.extend([r.model_dump() for r in results])
    
    return {
        **state,
        "search_results": new_results,
        "status": "searching",
        "progress_messages": [f"🔍 Searched: {current_q[:50]}..."],
    }


async def analyze_results(state: ResearchState) -> ResearchState:
    """Extract key findings from search results."""
    llm = get_llm()
    idx = state["current_question_idx"]
    current_q = state["sub_questions"][idx]
    
    # Get recent results
    recent_results = state["search_results"][-5:]
    formatted = format_results_for_llm([SearchResult(**r) for r in recent_results])
    
    prompt = f"""Based on these search results, extract 3-5 key factual findings that answer this question.
    
Question: {current_q}

Search Results:
{formatted}

Output a brief bulleted list of findings. Cite sources using [1], [2], etc."""

    response = await llm.ainvoke(prompt)
    
    new_findings = state["findings"].copy()
    new_findings.append(f"## {current_q}\n{response.content}")
    
    return {
        **state,
        "findings": new_findings,
        "current_question_idx": idx + 1,
        "iteration": state["iteration"] + 1,
        "status": "analyzing",
        "progress_messages": [f"📊 Analyzed findings for question {idx + 1}"],
    }


async def synthesize_answer(state: ResearchState) -> ResearchState:
    """Synthesize final comprehensive answer."""
    llm = get_llm()
    
    all_findings = "\n\n".join(state["findings"])
    sources = [SearchResult(**r) for r in state["search_results"]]
    source_list = "\n".join([f"[{i+1}] {s.title} - {s.url}" for i, s in enumerate(sources[:20])])
    
    prompt = f"""Based on all the research findings, write a comprehensive answer to the original question.

Original Question: {state["query"]}

Research Findings:
{all_findings}

Sources:
{source_list}

Format your answer with:
1. **Executive Summary** - 2-3 sentences
2. **Key Findings** - Main points with citations [1], [2], etc.
3. **Conclusion** - Brief takeaways

Use markdown formatting."""

    response = await llm.ainvoke(prompt)
    
    return {
        **state,
        "final_answer": response.content,
        "status": "complete",
        "progress_messages": [f"✅ Research complete!"],
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
    
    # Add nodes
    workflow.add_node("plan", plan_research)
    workflow.add_node("search", search_web)
    workflow.add_node("analyze", analyze_results)
    workflow.add_node("synthesize", synthesize_answer)
    
    # Add edges
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


# Create the compiled graph
research_graph = create_research_graph()


async def run_research(query: str, on_progress=None):
    """
    Run the research agent with progress callbacks.
    
    Args:
        query: The research question
        on_progress: Optional callback for progress updates
        
    Returns:
        Final research state with answer
    """
    initial_state: ResearchState = {
        "query": query,
        "sub_questions": [],
        "current_question_idx": 0,
        "search_results": [],
        "findings": [],
        "final_answer": "",
        "iteration": 0,
        "max_iterations": 10,
        "status": "starting",
        "progress_messages": [],
    }
    
    # Run with streaming
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
                    })
    
    # Get final state
    final_state = await research_graph.ainvoke(initial_state)
    return final_state
