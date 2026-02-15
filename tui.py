"""
Socrates & Plato — Terminal User Interface

Full-screen Textual TUI for the generator-critic agent loop.
Launch: uv run python cli.py
"""

import asyncio
import os
from pathlib import Path

import httpx
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, Container
from textual.reactive import reactive
from textual.widgets import (
    Header,
    Footer,
    Static,
    Input,
    Button,
    TextArea,
    ProgressBar,
    Markdown,
    Label,
    Rule,
    Select,
    Switch,
)

from config import set_backend, get_backend, SEARXNG_BASE_URL


# ─── Styled Widgets ──────────────────────────────────────────────────────────


class AgentPanel(Vertical):
    """A scrollable panel for one agent's output."""

    DEFAULT_CSS = """
    AgentPanel {
        border: solid $surface-lighten-2;
        height: 1fr;
        overflow-y: auto;
        padding: 1 2;
    }
    """

    def __init__(self, title: str, border_color: str, **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._border_color = border_color

    def compose(self) -> ComposeResult:
        yield Label(self._title, classes="panel-title")
        yield Markdown("", id=f"{self.id}-content")

    def update_content(self, content: str):
        md = self.query_one(f"#{self.id}-content", Markdown)
        md.update(content)
        md.scroll_end(animate=False)

    def on_mount(self):
        self.styles.border = ("solid", self._border_color)


class StatusLine(Static):
    """Status bar at the bottom of the run view."""

    DEFAULT_CSS = """
    StatusLine {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }
    """


# ─── Main App ─────────────────────────────────────────────────────────────────


class SocratesApp(App):
    """Socrates & Plato — Generator-Critic Agent System"""

    TITLE = "SOCRATES & PLATO"
    SUB_TITLE = "Generator-Critic Agent System"

    CSS = """
    Screen {
        background: $surface-darken-1;
    }

    /* ── Input Screen ── */
    #input-screen {
        align: center middle;
        width: 100%;
        height: 100%;
        padding: 2 4;
    }

    #input-box {
        width: 80%;
        max-width: 100;
        height: auto;
        align: center middle;
        padding: 2 3;
        border: tall $primary;
        background: $surface;
    }

    #input-box Label {
        margin-bottom: 1;
        text-style: bold;
        color: $text;
    }

    #task-input {
        height: 10;
        margin-bottom: 1;
    }

    #controls {
        height: 3;
        align: center middle;
    }

    #controls Label {
        margin: 0 1;
        text-style: bold;
    }

    #iter-input {
        width: 8;
        margin: 0 1;
    }

    #output-input {
        width: 20;
        margin: 0 1;
    }

    #backend-select {
        width: 18;
        margin: 0 1;
    }

    #run-btn {
        margin-left: 2;
        min-width: 16;
        background: $success;
        color: $text;
        text-style: bold;
    }

    #run-btn:hover {
        background: $success-lighten-1;
    }

    .subtitle-label {
        color: $text-muted;
        text-style: italic;
        margin-bottom: 1;
        text-align: center;
        width: 100%;
    }

    /* ── Run Screen ── */
    #run-screen {
        display: none;
        width: 100%;
        height: 100%;
    }

    #progress-area {
        dock: top;
        height: 3;
        padding: 0 2;
        background: $surface;
        align: left middle;
    }

    #progress-label {
        text-style: bold;
        margin-right: 2;
        width: auto;
        color: $primary-lighten-2;
    }

    #progress-bar {
        width: 1fr;
    }

    #panels {
        height: 1fr;
    }

    .panel-title {
        text-style: bold;
        margin-bottom: 1;
        text-align: center;
        width: 100%;
    }

    #socrates-panel .panel-title {
        color: #50fa7b;
    }

    #plato-panel .panel-title {
        color: #f1fa8c;
    }

    #socrates-panel {
        width: 3fr;
    }

    #plato-panel {
        width: 2fr;
    }

    #status-line {
        dock: bottom;
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }

    /* ── Done overlay ── */
    #done-banner {
        display: none;
        dock: bottom;
        height: 3;
        background: $success;
        color: $text;
        text-style: bold;
        text-align: center;
        padding: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", show=True),
        Binding("ctrl+c", "quit", "Abort", show=False),
        Binding("escape", "focus_task", "Focus Task", show=False),
    ]

    current_iteration: reactive[int] = reactive(0)
    max_iterations: reactive[int] = reactive(3)
    is_running: reactive[bool] = reactive(False)
    searxng_available: bool = False

    def compose(self) -> ComposeResult:
        yield Header()

        # ── Input Screen ──
        with Container(id="input-screen"):
            with Vertical(id="input-box"):
                yield Label("⚡ Enter your research task")
                yield Static(
                    "Socrates will generate, Plato will critique, and they'll iterate until perfection.",
                    classes="subtitle-label",
                )
                yield TextArea(id="task-input")
                yield Label("📄 Context File (optional):")
                yield Input(id="file-input", placeholder="/path/to/file.txt")
                yield Rule()
                with Horizontal(id="controls"):
                    yield Label("Iterations:")
                    yield Input(value="3", id="iter-input", type="integer")
                    yield Label("Output:")
                    yield Input(value="./output", id="output-input")
                    yield Label("Backend:")
                    yield Select(
                        [("Groq", "groq"), ("LMStudio", "lmstudio")],
                        value="groq",
                        id="backend-select",
                        allow_blank=False,
                    )
                    yield Button("▶  Run", id="run-btn", variant="success")

        # ── Run Screen ──
        with Vertical(id="run-screen"):
            with Horizontal(id="progress-area"):
                yield Label("● Starting...", id="progress-label")
                yield ProgressBar(total=100, show_eta=False, show_percentage=False, id="progress-bar")
            with Horizontal(id="panels"):
                yield AgentPanel("🏛  SOCRATES  —  Generator", "#50fa7b", id="socrates-panel")
                yield AgentPanel("📜  PLATO  —  Critic", "#f1fa8c", id="plato-panel")
            yield Static("Ready", id="status-line")
            yield Static("", id="done-banner")

        yield Footer()

    def on_mount(self):
        self.query_one("#task-input", TextArea).focus()
        self._check_searxng()

    @work(thread=False)
    async def _check_searxng(self):
        """Check if SearxNG is reachable on startup."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{SEARXNG_BASE_URL}/healthz", timeout=3.0)
                if resp.status_code == 200:
                    self.searxng_available = True
                    self.notify("🔍 SearxNG is reachable — web search enabled.", severity="information")
                else:
                    self.searxng_available = False
                    self.notify(
                        f"⚠️ SearxNG returned {resp.status_code}. Web search will be unavailable.",
                        severity="warning",
                    )
        except Exception:
            self.searxng_available = False
            self.notify(
                "⚠️ SearxNG is not reachable. The agent will run without web search.",
                severity="warning",
            )

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "run-btn" and not self.is_running:
            self._start_run()

    def action_focus_task(self):
        try:
            self.query_one("#task-input", TextArea).focus()
        except Exception:
            pass

    def _start_run(self):
        task_area = self.query_one("#task-input", TextArea)
        task_text = task_area.text.strip()
        file_path_input = self.query_one("#file-input", Input).value.strip()

        if not task_text:
            self.notify("Please enter a task first!", severity="error")
            return

        # Set backend from select dropdown
        try:
            backend_val = self.query_one("#backend-select", Select).value
            set_backend(str(backend_val) if backend_val else "groq")
        except Exception:
            set_backend("groq")

        # Warn about SearxNG if unreachable
        if not self.searxng_available:
            self.notify(
                "⚠️ Running without web search (SearxNG unavailable). "
                "The agent will use its training data only.",
                severity="warning",
            )

        # Check if task is a file path (legacy support)
        if os.path.isfile(task_text):
            task_text = Path(task_text).read_text().strip()

        # Validate file input
        file_path = None
        if file_path_input:
            if os.path.isfile(file_path_input):
                file_path = file_path_input
            else:
                self.notify(f"File not found: {file_path_input}", severity="warning")
                file_path = None

        try:
            iters = int(self.query_one("#iter-input", Input).value)
            if iters < 1:
                iters = 1
        except ValueError:
            iters = 3

        output_dir = self.query_one("#output-input", Input).value.strip() or "./output"

        self.max_iterations = iters
        self.current_iteration = 0
        self.is_running = True

        # Switch screens
        self.query_one("#input-screen").styles.display = "none"
        self.query_one("#run-screen").styles.display = "block"

        # Reset panels
        backend_label = get_backend().upper()
        search_status = "🔍" if self.searxng_available else "⚠️ no search"
        self.query_one("#socrates-panel", AgentPanel).update_content(
            f"*Waiting for first generation... ({backend_label} · {search_status})*"
        )
        self.query_one("#plato-panel", AgentPanel).update_content("*Waiting for critique...*")
        self.query_one("#progress-bar", ProgressBar).update(total=iters, progress=0)
        self.query_one("#progress-label").update(f"● Iteration 0 / {iters}")
        self.query_one("#status-line").update(f"⏳ Starting agent loop... [{backend_label}]")
        self.query_one("#done-banner").styles.display = "none"

        self._run_agents(task_text, output_dir, iters, file_path)

    @work(thread=False)
    async def _run_agents(self, task: str, output_dir: str, iterations: int, file_path: str | None):
        from graph import run_task

        async def on_event(event_type: str, data: dict):
            if event_type == "start":
                self.query_one("#status-line").update(
                    f"🔍 Task: {data['task'][:80]}..."
                )

            elif event_type == "generate":
                iteration = data["iteration"]
                self.current_iteration = iteration
                self.query_one("#progress-label").update(
                    f"● Iteration {iteration} / {self.max_iterations}  —  Socrates is writing..."
                )
                self.query_one("#progress-bar", ProgressBar).update(
                    progress=max(0, iteration - 1)
                )

                response = data.get("response", "")
                search = data.get("search_context", "")

                content = f"### Iteration {iteration}\n\n{response}"
                if search and search not in ("Search unavailable.", "No search results found."):
                    content += f"\n\n---\n\n*🔍 Search context used*"

                self.query_one("#socrates-panel", AgentPanel).update_content(content)

                if search and search not in ("Search unavailable.", "No search results found."):
                    self.query_one("#status-line").update(f"🔍 Web search completed for iteration {iteration}")
                else:
                    self.query_one("#status-line").update(f"✍️  Socrates generated response for iteration {iteration}")

            elif event_type == "critique":
                iteration = data["iteration"]
                self.query_one("#progress-label").update(
                    f"● Iteration {iteration} / {self.max_iterations}  —  Plato is reviewing..."
                )

                feedback = data.get("feedback", "")
                self.query_one("#plato-panel", AgentPanel).update_content(
                    f"### Iteration {iteration}\n\n{feedback}"
                )
                self.query_one("#status-line").update(f"📜 Plato critiqued iteration {iteration}")

            elif event_type == "save":
                path = data.get("path", "")
                self.query_one("#status-line").update(f"💾 Saved {path}")
                self.query_one("#progress-bar", ProgressBar).update(
                    progress=data.get("iteration", 0)
                )

            elif event_type == "done":
                self.query_one("#progress-bar", ProgressBar).update(
                    progress=self.max_iterations
                )
                self.query_one("#progress-label").update(
                    f"✓ Complete  —  {self.max_iterations} iterations"
                )
                path = data.get("path", "")
                self.query_one("#status-line").update(f"✅ Final output saved to {path}")
                self.query_one("#done-banner").update(
                    f"✓ Done! Output saved to {path}  ·  Press [q] to quit"
                )
                self.query_one("#done-banner").styles.display = "block"

                # Update socrates panel with final response
                final = data.get("final_response", "")
                self.query_one("#socrates-panel", AgentPanel).update_content(
                    f"## Final Response\n\n{final}"
                )

            elif event_type == "error":
                self.query_one("#status-line").update(f"❌ {data.get('message', 'Error')}")
                self.notify(data.get("message", "An error occurred"), severity="error")

        try:
            await run_task(
                task=task,
                output_dir=output_dir,
                iterations=iterations,
                file_path=file_path,
                on_event=on_event,
            )
        except Exception as e:
            self.query_one("#status-line").update(f"❌ Error: {e}")
            self.notify(f"Error: {e}", severity="error")
        finally:
            self.is_running = False


def launch_tui():
    """Entry point for launching the TUI."""
    app = SocratesApp()
    app.run()


if __name__ == "__main__":
    launch_tui()
