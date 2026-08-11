"""State definition for the debate LangGraph state machine."""

import operator
from typing import Annotated, TypedDict


class DebateState(TypedDict):
    topic: str
    rounds: int
    current_round: int
    research_notes: list[str]
    proponent_arguments: Annotated[list[str], operator.add]
    opponent_arguments: Annotated[list[str], operator.add]
    critiques: Annotated[list[str], operator.add]
    verdict: str
