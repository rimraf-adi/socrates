from typing import Literal, Callable, Awaitable
import asyncio


from langgraph.graph import StateGraph, END
from rich.console import Console

from models import AgentState
from generator import generate
from critic import critique
from logger import save_iteration, save_final, update_summary


console = Console()

EventCallback = Callable[[str, dict], Awaitable[None]] | None


def should_continue(state: AgentState) -> Literal["generate", "end"]:
    if state["iteration"] < state["max_iterations"]:
        return "generate"
    return "end"


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("generate", generate)
    workflow.add_node("critique", critique)

    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "critique")
    workflow.add_conditional_edges(
        "critique",
        should_continue,
        {
            "generate": "generate",
            "end": END,
        },
    )

    return workflow.compile()


graph = build_graph()


async def run_task(
    task: str,
    output_dir: str,
    iterations: int = 3,
    model: str | None = None,
    file_path: str | None = None,
    on_event: EventCallback = None,
) -> str:
    initial_state: AgentState = {
        "task": task,
        "current_response": "",
        "feedback": "",
        "iteration": 0,
        "max_iterations": iterations,
        "history": [],
        "final_response": "",
        "status": "starting",
        "search_context": "",
        "file_path": file_path,
    }

    async def emit(event_type: str, data: dict):
        if on_event:
            await on_event(event_type, data)

    await emit("start", {"task": task, "iterations": iterations, "output_dir": output_dir, "file_path": file_path})

    if not on_event:
        console.print(f"\n[bold cyan]Task:[/] {task}")
        console.print(f"[bold cyan]Iterations:[/] {iterations}")
        console.print(f"[bold cyan]Output:[/] {output_dir}\n")

    final_state = None

    try:
        async for event in graph.astream(initial_state):
            for node_name, node_state in event.items():
                if node_name == "generate":
                    current_iter = (node_state.get("iteration", 0) or 0) + 1
                    await emit("generate", {
                        "iteration": current_iter,
                        "response": node_state.get("current_response", ""),
                        "search_context": node_state.get("search_context", ""),
                    })
                    if not on_event:
                        console.print(f"[bold green]▶ Iteration {current_iter}:[/] Generator produced response")

                elif node_name == "critique":
                    current_iter = node_state.get("iteration", 0) or 0
                    feedback = ""
                    if node_state.get("history"):
                        latest = node_state["history"][-1]
                        feedback = latest.get("critic_feedback", "")
                        path = save_iteration(output_dir, latest)  # Keep saving individual iterations
                        
                        # New: Update summary with global context
                        update_summary(output_dir, task, node_state["history"])
                        
                        await emit("save", {"path": path, "iteration": current_iter})
                        if not on_event:
                            console.print(f"  [dim]→ Saved {path}[/]")

                    await emit("critique", {
                        "iteration": current_iter,
                        "feedback": feedback,
                    })
                    if not on_event:
                        console.print(f"[bold yellow]◆ Iteration {current_iter}:[/] Critic provided feedback")

                final_state = node_state

    except (KeyboardInterrupt, asyncio.CancelledError):
        await emit("error", {"message": "Process interrupted. Saving progress..."})
        if not on_event:
            console.print("\n[bold red]Interrupted! Saving current state...[/]")
        # We don't break here, we just fall through to the final state check
    except Exception as e:
        await emit("error", {"message": f"Error: {e}"})
        if not on_event:
            console.print(f"\n[bold red]Error: {e}[/]")


    if final_state is None:
        await emit("error", {"message": "No output produced (or interrupted early)."})
        if not on_event:
            console.print("[bold red]No output produced.[/]")
        return ""


    final_response = final_state.get("current_response", "")
    history = final_state.get("history", [])

    final_path = save_final(output_dir, task, history, final_response)
    await emit("done", {"final_response": final_response, "path": final_path})

    if not on_event:
        console.print(f"\n[bold green]✓ Final output saved to {final_path}[/]")

    return final_response
