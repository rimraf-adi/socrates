from config import get_llm, truncate
from models import AgentState
from tools import search_web


GENERATOR_INITIAL_PROMPT = """You are an expert content generator. You have been given a task to complete.

Task: {task}

{search_section}

Generate a comprehensive, high-quality response to this task. Be thorough and detailed."""

GENERATOR_REFINE_PROMPT = """You are an expert content generator. You previously generated a response and received feedback from a critic.

Task: {task}

Your previous response:
{previous_response}

Critic's feedback:
{feedback}

{search_section}

Improve your response based on the critic's feedback. Address every point raised. Generate the complete improved response."""


async def generate(state: AgentState) -> dict:
    llm = get_llm(temperature=0.5)
    iteration = state["iteration"]

    search_results = ""
    try:
        search_query = state["task"][:200]
        search_results = await search_web(search_query, max_results=3)
    except Exception:
        search_results = ""

    search_section = ""
    if search_results and search_results != "Search unavailable.":
        search_section = f"Web research context:\n{truncate(search_results, 4000)}"

    if iteration == 0:
        prompt = GENERATOR_INITIAL_PROMPT.format(
            task=truncate(state["task"], 2000),
            search_section=search_section,
        )
    else:
        prompt = GENERATOR_REFINE_PROMPT.format(
            task=truncate(state["task"], 2000),
            previous_response=truncate(state["current_response"], 10000),
            feedback=truncate(state["feedback"], 8000),
            search_section=search_section,
        )

    response = await llm.ainvoke(prompt)

    return {
        "current_response": response.content,
        "search_context": search_results,
        "status": "generated",
    }

