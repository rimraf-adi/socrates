import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Generator-Critic Multi-Agent System")
console = Console()


@app.command()
def run(
    task: str = typer.Option(..., "--task", "-t", help="Task description or path to a .md/.txt file"),
    output: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    iterations: int = typer.Option(3, "--iterations", "-n", help="Number of generator-critic iterations"),
    model: str = typer.Option(None, "--model", "-m", help="LMStudio model override"),
):
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
