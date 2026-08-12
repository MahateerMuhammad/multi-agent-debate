"""Typed debate state representation for LangGraph workflow orchestration."""

from __future__ import annotations

from typing import Any, TypedDict


class DebateState(TypedDict, total=False):
    """Typed state dictionary passed through the LangGraph debate workflow."""

    topic: str
    current_round: int
    max_rounds: int
    confidence_threshold: float
    improvement_threshold: float
    proponent_history: list[dict[str, Any]]
    opponent_history: list[dict[str, Any]]
    critic_history: list[dict[str, Any]]
    judge_history: list[dict[str, Any]]
    is_finished: bool
    stop_reason: str
    total_latency: float
    total_tokens: int
    errors: list[str]
