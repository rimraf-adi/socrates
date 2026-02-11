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
To use a tool, you MUST use this format:

Thought: ... reasoning about what to do next ...
Action: tool_name(arg1="value", arg2=123)
Observation: ... (this will be filled by the system)

When you have enough information, provide your final answer:
Final Answer: ... your detailed response ...
"""

TASK_PROMPT = """
Task: {task}
File Path: {file_path_info}

{feedback_section}

Begin!
"""

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

Refine your previous response based on the feedback.
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
    max_steps = 5
    current_search_context = [] # Track what we found for the logs
    
    # Initial thought
    response_text = ""
    
    for step in range(max_steps):
        # Call LLM
        # We need to construct the full string history because ChatOpenAI generic usage 
        # is easier with a single prompt if we are manually doing ReAct, 
        # but let's try maintaining the messages list for chat models.
        
        # Determine strictness? LMStudio models vary.
        # Let's just create a string prompt from messages for simplicity if needed,
        # or pass the messages list if supported. 
        # LangChain's ChatOpenAI handles checking.
        
        # Actually, let's keep it simple: Append to prompt string.
        prompt_str = "\n".join([m[1] for m in messages])
        
        response = await llm.ainvoke(prompt_str)
        content = response.content
        
        messages.append(("assistant", content))
        response_text = content
        
        # Parse for Action
        # Regex to find: Action: tool_name(args)
        # We look for the LAST action in the block if multiple (rare).
        action_match = re.search(r"Action:\s*(\w+)\((.*)\)", content, re.DOTALL)
        
        if "Final Answer:" in content:
            # We are done. extracting final answer.
            final_ans = content.split("Final Answer:")[-1].strip()
            return {
                "current_response": final_ans,
                "search_context": "\n".join(current_search_context),
                "status": "generated",
            }
        
        if not action_match:
            # No action, no final answer? 
            # If the model just talks without following format, we treat it as final answer if it looks substantial,
            # or we prompt it to format.
            # For now, let's assume it IS the answer if it's long, or force it.
            if len(content) > 200:
                 return {
                    "current_response": content,
                    "search_context": "\n".join(current_search_context),
                    "status": "generated",
                }
            else:
                 messages.append(("user", "Please continue. If finished, say 'Final Answer:'."))
                 continue

        # Execute Tool
        tool_name = action_match.group(1).strip()
        args_str = action_match.group(2).strip()
        
        observation = f"Error: Tool {tool_name} not found."
        
        try:
            # Parse args - simple heuristic or eval (careful!)
            # We'll try to parse key=value or just positional if simple.
            # Using eval is risky but for a local agent tool it's effective for parsing python-like args args_str
            # We will use valid python eval but capture errors.
            # Security risk? It's a local agent running user code.
            # Let's try to be safer: simplistic parsing.
            
            # Reconstruct args:
            # tool(query="foo") -> query="foo"
            
            # Let's just use `eval`.
            # Restrict globals/locals for safety.
            allowed_names = {"True": True, "False": False, "None": None}
            
            # Simple check
            if tool_name == "search_web":
                # regex extract query
                # pattern: query="(.*)" or query='(.*)' or just the string?
                # Let's assume the LLM follows `search_web(query="...")`
                match = re.search(r'query=["\'](.*?)["\']', args_str)
                if match:
                    q = match.group(1)
                    res = await search_web(q)
                    observation = f"Search Results:\n{truncate(res, 2000)}"
                    current_search_context.append(f"Query: {q}\n{res}")
                else:
                    # try positional: search_web("...")
                    match = re.search(r'["\'](.*?)["\']', args_str)
                    if match:
                        q = match.group(1)
                        res = await search_web(q)
                        observation = f"Search Results:\n{truncate(res, 2000)}"
                        current_search_context.append(f"Query: {q}\n{res}")
                    else:
                        observation = "Error processsing arguments. Usage: search_web(query='...')"

            elif tool_name == "read_file":
                # usage: read_file(file_path="...", start_line=1, end_line=10)
                # Parse args using eval in a controlled env because parsing int/str is annoying with regex
                # We trust the local LLM outputs for this prototype
                
                # Mock function to capture args
                def _mock_read(file_path, start_line=None, end_line=None):
                    return read_file(file_path, start_line, end_line)
                
                try:
                    # We wrap the call in a lambda to eval it
                    # "read_file(...)" is the string.
                    # We need to eval `_mock_read(...)`
                    # Reconstruct the string to use our local function name
                   
                    # This is tricky because `tool_name` is "read_file".
                    # We can map it.
                    
                    local_tools = {
                        "read_file": _mock_read,
                        "search_web": None # Handled above
                    }
                    
                    # Construct expression
                    expr = f"{tool_name}({args_str})"
                    
                    if tool_name == "read_file":
                         # Disable builtins for safety
                        observation = eval(expr, {"__builtins__": {}}, local_tools)
                        observation = f"File Content:\n{truncate(observation, 3000)}"
                    
                except Exception as e:
                    observation = f"Error executing read_file: {e}"

        except Exception as e:
            observation = f"Error parsing action: {e}"

        # Feed back observation
        messages.append(("user", f"Observation: {observation}"))
        
    # If partial response, return it
    return {
        "current_response": response_text,
        "search_context": "\n".join(current_search_context),
        "status": "generated",
    }
