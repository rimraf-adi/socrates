import os
from models import IterationRecord


def save_iteration(output_dir: str, record: IterationRecord) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"iteration_{record['iteration']:02d}.md"
    filepath = os.path.join(output_dir, filename)

    content = f"""# Iteration {record['iteration']}

## Generator Response

{record['generator_response']}

---

## Critic Feedback

{record['critic_feedback']}
"""
    with open(filepath, "w") as f:
        f.write(content)

    return filepath


def save_final(output_dir: str, task: str, history: list[IterationRecord], final_response: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "final.md")

    sections = [f"# Task\n\n{task}\n"]

    for record in history:
        sections.append(f"""---

# Iteration {record['iteration']}

## Generator Response

{record['generator_response']}

## Critic Feedback

{record['critic_feedback']}
""")

    sections.append(f"""---

# Final Response

{final_response}
""")

    with open(filepath, "w") as f:
        f.write("\n".join(sections))

    return filepath
