import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Socrates & Plato — Generator-Critic Agent System")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    task: str = typer.Option(None, "--task", "-t", help="Task description or path to a .md/.txt file"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    iterations: int = typer.Option(3, "--iterations", "-n", help="Number of generator-critic iterations"),
    model: str = typer.Option(None, "--model", "-m", help="LMStudio model override"),
):
    """Launch TUI (default) or run headless with --task."""
    if task is None:
        # No task flag → launch TUI
        from tui import launch_tui
        launch_tui()
        return

    # Headless CLI mode
    task_text = task
    if os.path.isfile(task):
        task_text = Path(task).read_text().strip()

    from graph import run_task

    result = asyncio.run(run_task(
        task=task_text,
        output_dir=output,
        iterations=iterations,
        model=model,
    ))

    if result:
        console.print("\n[bold]═══ Final Response ═══[/]\n")
        console.print(result)


if __name__ == "__main__":
    app()
