from config import get_llm
from models import AgentState


CRITIC_PROMPT = """You are an expert critic and reviewer. Your job is to rigorously evaluate a response and provide actionable feedback.

Original task: {task}

Response to review (iteration {iteration}):
{response}

Evaluate this response on:
1. **Accuracy** — Are the claims correct and well-supported?
2. **Completeness** — Does it fully address the task?
3. **Clarity** — Is it well-structured and easy to follow?
4. **Depth** — Does it go beyond surface-level treatment?
5. **Actionability** — Are suggestions/conclusions practical?

Provide your feedback in this format:

## Strengths
- (list what works well)

## Weaknesses
- (list specific problems)

## Suggestions for Improvement
- (list concrete, actionable changes)

## Overall Assessment
(brief summary judgment)"""


async def critique(state: AgentState) -> dict:
    llm = get_llm(temperature=0.3)
    iteration = state["iteration"]

    prompt = CRITIC_PROMPT.format(
        task=state["task"],
        iteration=iteration + 1,
        response=state["current_response"],
    )

    response = await llm.ainvoke(prompt)

    history = state["history"].copy()
    history.append({
        "iteration": iteration + 1,
        "generator_response": state["current_response"],
        "critic_feedback": response.content,
    })

    return {
        "feedback": response.content,
        "iteration": iteration + 1,
        "history": history,
        "status": "critiqued",
    }
