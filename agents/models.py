from typing import TypedDict, List


class IterationRecord(TypedDict):
    iteration: int
    generator_response: str
    critic_feedback: str


class AgentState(TypedDict):
    task: str
    current_response: str
    feedback: str
    iteration: int
    max_iterations: int
    history: List[IterationRecord]
    final_response: str
    status: str
    search_context: str
