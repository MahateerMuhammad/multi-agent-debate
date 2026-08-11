"""Graph node functions for LangGraph execution."""

from app.graph.state import DebateState


def research_node(state: DebateState) -> dict:
    return {"research_notes": [f"Initial context for: {state.get('topic', '')}"]}


def proponent_node(state: DebateState) -> dict:
    return {"proponent_arguments": [f"Round {state.get('current_round', 1)} Proponent Argument"]}


def opponent_node(state: DebateState) -> dict:
    return {"opponent_arguments": [f"Round {state.get('current_round', 1)} Opponent Rebuttal"]}


def critic_node(state: DebateState) -> dict:
    return {"critiques": [f"Round {state.get('current_round', 1)} Critique"]}


def judge_node(state: DebateState) -> dict:
    return {"verdict": "Debate concluded with verdict."}
