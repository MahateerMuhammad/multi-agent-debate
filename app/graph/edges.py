"""Conditional routing logic and deterministic stopping edge functions for LangGraph."""

from __future__ import annotations

from typing import Literal

from app.graph.state import DebateState


def should_continue(state: DebateState) -> Literal["proponent", "end"]:
    """Evaluate 4 deterministic criteria to route back to Proponent or terminate graph."""
    errors = state.get("errors", [])
    if errors:
        return "end"

    current_round = state.get("current_round", 0)
    max_rounds = state.get("max_rounds", 3)
    if current_round >= max_rounds:
        return "end"

    judge_history = state.get("judge_history", [])
    if not judge_history:
        return "proponent"

    latest_judge = judge_history[-1]
    score_a = latest_judge.get("total_score_a", 0.0)
    score_b = latest_judge.get("total_score_b", 0.0)
    max_score = max(score_a, score_b)

    # Criterion 1: Confidence threshold met
    conf_threshold = state.get("confidence_threshold", 0.90)
    if max_score >= conf_threshold:
        return "end"

    # Criterion 2: Score improvement threshold check (multi-round)
    if len(judge_history) >= 2:
        prev_judge = judge_history[-2]
        prev_score = max(prev_judge.get("total_score_a", 0.0), prev_judge.get("total_score_b", 0.0))
        improvement = max_score - prev_score
        imp_threshold = state.get("improvement_threshold", 0.02)
        if improvement < imp_threshold:
            return "end"

    return "proponent"
