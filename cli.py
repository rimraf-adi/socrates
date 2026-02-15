import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console

from config import set_backend

app = typer.Typer(help="Socrates & Plato — Generator-Critic Agent System")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    task: str = typer.Option(None, "--task", "-t", help="Task description or path to a .md/.txt file"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    iterations: int = typer.Option(3, "--iterations", "-n", help="Number of generator-critic iterations"),
    model: str = typer.Option(None, "--model", "-m", help="Model override"),
    file_path: str = typer.Option(None, "--file", "-f", help="Context file path for agentic reading"),
    backend: str = typer.Option("groq", "--backend", "-b", help="LLM backend: 'groq' or 'lmstudio'"),
):
    """Launch TUI (default) or run headless with --task."""
    # Set backend before anything else
    set_backend(backend)

    if task is None:
        # No task flag → launch TUI
        from tui import launch_tui
        launch_tui()
        return

    # Headless CLI mode
    from graph import run_task
    from config import SEARXNG_BASE_URL
    import httpx

    task_text = task
    if os.path.isfile(task):
        task_text = Path(task).read_text().strip()

    # Check SearxNG reachability
    try:
        resp = httpx.get(f"{SEARXNG_BASE_URL}/healthz", timeout=3.0)
        if resp.status_code == 200:
            console.print("[green]🔍 SearxNG is reachable — web search enabled.[/]")
        else:
            console.print(f"[yellow]⚠️  SearxNG returned {resp.status_code}. Web search may be unavailable.[/]")
    except Exception:
        console.print("[yellow]⚠️  SearxNG is not reachable. Running without web search.[/]")
        if not typer.confirm("Continue without web search?", default=True):
            raise typer.Exit()

    result = asyncio.run(run_task(
        task=task_text,
        output_dir=output,
        iterations=iterations,
        model=model,
        file_path=file_path,
    ))

    if result:
        console.print("\n[bold]═══ Final Response ═══[/]\n")
        console.print(result)


if __name__ == "__main__":
    app()
