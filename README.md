# Socrates & Plato — Generator-Critic Agent System

Two-agent iterative refinement system. **Socrates** (generator) produces responses with web search via SearXNG, **Plato** (critic) reviews and provides feedback, and the loop repeats for N iterations.

## Architecture

```
Task → Generator (+ SearXNG search) → Critic → [repeat N times] → Final Output
```

- **Generator (Socrates)**: receives task + critic feedback, searches the web for context, produces a response
- **Critic (Plato)**: evaluates the response on accuracy, completeness, clarity, depth, and actionability
- **Logger**: saves each iteration and the full conversation to markdown files

Built with [LangGraph](https://github.com/langchain-ai/langgraph) and [LMStudio](https://lmstudio.ai/) LLMs.

## Setup

```bash
# 1. Start SearXNG
docker-compose up

# 2. Install dependencies
uv sync

# 3. Start LMStudio with a model loaded
```

Requires:
- **LMStudio** running at `http://127.0.0.1:1234` with a model loaded
- **SearXNG** running at `http://localhost:8080` (via `docker-compose up`)

## Usage

```bash
uv run python cli.py -t "Write an essay about quantum computing" -o ./output -n 3
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-t, --task` | Task description or path to a `.md`/`.txt` file | **required** |
| `-o, --output` | Output directory | `./output` |
| `-n, --iterations` | Number of generator-critic iterations | `3` |
| `-m, --model` | LMStudio model override | auto |

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LMSTUDIO_BASE_URL` | LMStudio API endpoint | `http://127.0.0.1:1234/v1` |
| `SEARXNG_BASE_URL` | SearXNG instance URL | `http://localhost:8080` |

## Output Structure

```
output/
├── iteration_01.md    # Generator response + Critic feedback (round 1)
├── iteration_02.md    # Generator response + Critic feedback (round 2)
├── iteration_03.md    # Generator response + Critic feedback (round 3)
└── final.md           # Complete conversation log + final response
```

## Project Structure

```
socrates/
├── docker-compose.yml  # SearXNG container
├── searxng/            # SearXNG configuration
├── pyproject.toml      # Dependencies
├── config.py           # LLM and SearXNG configuration
├── models.py           # State and data models
├── tools.py            # SearXNG web search tool
├── generator.py        # Generator agent (Socrates)
├── critic.py           # Critic agent (Plato)
├── graph.py            # LangGraph orchestration
├── logger.py           # Markdown file logging
├── cli.py              # CLI entry point
└── README.md           # This file
```
