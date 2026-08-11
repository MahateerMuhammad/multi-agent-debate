"""State definition for the debate LangGraph state machine."""

from typing import TypedDict, List, Annotated
import operator

class DebateState(TypedDict):
    topic: str
    rounds: int
    current_round: int
    research_notes: List[str]
    proponent_arguments: Annotated[List[str], operator.add]
    opponent_arguments: Annotated[List[str], operator.add]
    critiques: Annotated[List[str], operator.add]
    verdict: str
