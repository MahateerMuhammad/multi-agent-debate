"""Typed debate state representation for LangGraph workflow orchestration."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from app.graph.schemas import StopReason


class DebateState(TypedDict, total=False):
    """Typed state dictionary passed through the LangGraph debate workflow."""

    topic: str
    current_round: int
    max_rounds: int
    confidence_threshold: float
    improvement_threshold: float
    proponent_history: list[dict[str, Any]]
    opponent_history: list[dict[str, Any]]
    evidence_history: list[dict[str, Any]]
    critic_history: list[dict[str, Any]]
    judge_history: list[dict[str, Any]]
    is_finished: bool
    stop_reason: StopReason
    total_latency: Annotated[float, operator.add]
    prompt_tokens: Annotated[int, operator.add]
    completion_tokens: Annotated[int, operator.add]
    total_tokens: Annotated[int, operator.add]
    llm_call_traces: Annotated[list[dict[str, Any]], operator.add]
    errors: list[str]
