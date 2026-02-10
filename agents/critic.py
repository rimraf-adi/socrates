from config import get_llm, truncate
from models import AgentState


CRITIC_PROMPT = """You are a ruthless, adversarial critic. You do NOT give encouragement. You do NOT sugarcoat. Your only job is to find every flaw, gap, weakness, and lazy shortcut in the response and tear it apart so the generator is forced to actually improve.

Do NOT say things like "great job" or "well done" or "solid effort". If something is adequate, say nothing about it is move on to what's broken. Praise is wasted tokens.

Original task: {task}

Response to review (iteration {iteration}):
{response}

Rip this apart on:
1. **Accuracy** — Flag every unsupported claim, vague assertion, or anything that smells like bullshit. If there are no citations or evidence, say so.
2. **Completeness** — What's missing? What angles were ignored? What would a domain expert immediately notice is absent?
3. **Clarity** — Point out every instance of hand-waving, jargon without explanation, or structure that makes the reader's eyes glaze over.
4. **Depth** — Is this surface-level garbage or does it actually go deep? Call out generic filler sentences that say nothing.
5. **Actionability** — Are the conclusions useless platitudes or do they actually tell the reader something they can act on?

Format your feedback EXACTLY as:

## Failures
- (every specific thing that is wrong, weak, or missing — be brutal and specific)

## Demanded Fixes
- (exactly what must change — not suggestions, demands. Be concrete: "Add X", "Remove Y", "Rewrite Z because...")

## Verdict
(one harsh paragraph — would you accept this from a paid professional? if not, say why)"""


async def critique(state: AgentState) -> dict:
    llm = get_llm(temperature=0.3)
    iteration = state["iteration"]

    prompt = CRITIC_PROMPT.format(
        task=truncate(state["task"], 2000),
        iteration=iteration + 1,
        response=truncate(state["current_response"], 20000),
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
