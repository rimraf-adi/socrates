import re
import json

from config import get_llm, truncate
from models import AgentState
from tools import search_web, read_file


SYSTEM_PROMPT = """You are Socrates, a deep-thinking analytical agent.
Your goal is to complete the user's task by gathering information and reasoning thoroughly.

You have access to the following tools:

1. search_web(query)
   - Search the internet for current information.
   - Use this when you need external facts not in your training data.

2. read_file(file_path, start_line=None, end_line=None)
   - Read content from a file.
   - You MUST specify start_line and end_line to read only what you need (e.g., 1-100 for intro) unless the file is very small.
   - Use this to inspect the file content provided in the task.

FORMAT INSTRUCTIONS:
To use a tool, you MUST use EXACTLY this format on a SINGLE LINE:

Thought: ... reasoning about what to do next ...
Action: tool_name(arg1="value", arg2=123)

The system will respond with:
Observation: ... result ...

When you have enough information, provide your final answer:
Final Answer: ... your detailed response ...

IMPORTANT:
- Put each Action on its OWN line.
- Do NOT split an Action across multiple lines.
- Always wrap string arguments in quotes.
"""

TASK_PROMPT = """
Task: {task}
File Path: {file_path_info}

{feedback_section}

Begin!
"""


def _parse_action(text: str):
    """Parse the LAST Action: line from model output. Returns (tool_name, args_str) or None."""
    # Match single-line Action: calls only (re.MULTILINE makes ^ match line starts)
    matches = re.findall(r'^Action:\s*(\w+)\((.+)\)\s*$', text, re.MULTILINE)
    if matches:
        return matches[-1]  # Take the last action if multiple
    return None


def _parse_search_args(args_str: str) -> str | None:
    """Extract query string from search_web args."""
    # Try query="..." or query='...'
    match = re.search(r'query\s*=\s*["\'](.+?)["\']', args_str)
    if match:
        return match.group(1)
    # Try positional: "..."
    match = re.search(r'["\'](.+?)["\']', args_str)
    if match:
        return match.group(1)
    # Try bare string (no quotes) — last resort
    stripped = args_str.strip().strip('"').strip("'")
    if stripped:
        return stripped
    return None


def _parse_read_file_args(args_str: str) -> tuple[str | None, int | None, int | None]:
    """Extract file_path, start_line, end_line from read_file args."""
    file_path = None
    start_line = None
    end_line = None

    # Extract file_path
    match = re.search(r'file_path\s*=\s*["\'](.+?)["\']', args_str)
    if match:
        file_path = match.group(1)
    else:
        # Positional: first quoted string
        match = re.search(r'["\'](.+?)["\']', args_str)
        if match:
            file_path = match.group(1)

    # Extract start_line
    match = re.search(r'start_line\s*=\s*(\d+)', args_str)
    if match:
        start_line = int(match.group(1))

    # Extract end_line
    match = re.search(r'end_line\s*=\s*(\d+)', args_str)
    if match:
        end_line = int(match.group(1))

    return file_path, start_line, end_line


async def generate(state: AgentState) -> dict:
    llm = get_llm(temperature=0.7)
    iteration = state["iteration"]
    task = state["task"]
    file_path = state.get("file_path", None)

    file_path_info = file_path if file_path else "No file provided."

    feedback_section = ""
    if iteration > 0:
        feedback_section = f"""
Previous Response:
{truncate(state['current_response'], 4000)}

Critic's Feedback:
{truncate(state['feedback'], 4000)}

You MUST address every point in the critic's feedback.
Expand sections the critic flagged as incomplete.
Add new content the critic demanded.
Do NOT just rephrase — add real substance.
"""

    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", TASK_PROMPT.format(
            task=truncate(task, 2000),
            file_path_info=file_path_info,
            feedback_section=feedback_section
        ))
    ]

    # ReAct Loop
    max_steps = 8
    current_search_context = []

    response_text = ""

    for step in range(max_steps):
        prompt_str = "\n".join([m[1] for m in messages])

        response = await llm.ainvoke(prompt_str)
        content = response.content

        messages.append(("assistant", content))
        response_text = content

        # Check for Final Answer first
        if "Final Answer:" in content:
            final_ans = content.split("Final Answer:")[-1].strip()
            return {
                "current_response": final_ans,
                "search_context": "\n".join(current_search_context),
                "status": "generated",
            }

        # Parse for Action using line-anchored regex
        action = _parse_action(content)

        if not action:
            # No action and no final answer
            if len(content) > 200:
                return {
                    "current_response": content,
                    "search_context": "\n".join(current_search_context),
                    "status": "generated",
                }
            else:
                messages.append(("user", "Please continue. Use 'Action: tool_name(...)' to use a tool, or 'Final Answer: ...' to give your response."))
                continue

        tool_name, args_str = action
        observation = f"Error: Tool '{tool_name}' not found. Available tools: search_web, read_file"

        try:
            if tool_name == "search_web":
                query = _parse_search_args(args_str)
                if query:
                    res = await search_web(query)
                    observation = f"Search Results:\n{truncate(res, 2000)}"
                    current_search_context.append(f"Query: {query}\n{res}")
                else:
                    observation = "Error: Could not parse query. Usage: search_web(query=\"your search query\")"

            elif tool_name == "read_file":
                fp, sl, el = _parse_read_file_args(args_str)
                if fp:
                    result = read_file(fp, sl, el)
                    observation = f"File Content:\n{truncate(result, 3000)}"
                else:
                    observation = "Error: Could not parse file_path. Usage: read_file(file_path=\"/path/to/file\", start_line=1, end_line=100)"

        except Exception as e:
            observation = f"Error executing {tool_name}: {e}"

        # Feed back observation
        messages.append(("user", f"Observation: {observation}"))

    # If we exhausted steps, return whatever we have
    return {
        "current_response": response_text,
        "search_context": "\n".join(current_search_context),
        "status": "generated",
    }
