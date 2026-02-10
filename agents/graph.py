from typing import Literal

from langgraph.graph import StateGraph, END
from rich.console import Console

from models import AgentState
from generator import generate
from critic import critique
from logger import save_iteration, save_final

console = Console()


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
    }

    console.print(f"\n[bold cyan]Task:[/] {task}")
    console.print(f"[bold cyan]Iterations:[/] {iterations}")
    console.print(f"[bold cyan]Output:[/] {output_dir}\n")

    final_state = None

    async for event in graph.astream(initial_state):
        for node_name, node_state in event.items():
            if node_name == "generate":
                current_iter = (node_state.get("iteration", 0) or 0) + 1
                console.print(f"[bold green]▶ Iteration {current_iter}:[/] Generator produced response")

            elif node_name == "critique":
                current_iter = node_state.get("iteration", 0) or 0
                console.print(f"[bold yellow]◆ Iteration {current_iter}:[/] Critic provided feedback")

                if node_state.get("history"):
                    latest = node_state["history"][-1]
                    path = save_iteration(output_dir, latest)
                    console.print(f"  [dim]→ Saved {path}[/]")

            final_state = node_state

    if final_state is None:
        console.print("[bold red]No output produced.[/]")
        return ""

    final_response = final_state.get("current_response", "")
    history = final_state.get("history", [])

    final_path = save_final(output_dir, task, history, final_response)
    console.print(f"\n[bold green]✓ Final output saved to {final_path}[/]")

    return final_response
