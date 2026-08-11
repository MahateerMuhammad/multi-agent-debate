"""Agent output schemas and debate container data models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.llm.schemas import (
    ArgumentOutput,
    CriticOutput,
    JudgeOutput,
    RebuttalOutput,
)


class SimpleDebateResult(BaseModel):
    """Container holding the results of a 2-agent Proponent vs Opponent debate."""

    topic: str = Field(..., description="The debate proposition topic")
    proponent_output: ArgumentOutput = Field(..., description="Proponent agent's argument")
    opponent_output: RebuttalOutput = Field(..., description="Opponent agent's rebuttal")
    total_latency_seconds: float = Field(..., description="Total elapsed latency for the debate")
    total_tokens: int = Field(..., description="Total token consumption across all calls")


class FullDebateResult(BaseModel):
    """Container holding the complete results of a 4-agent debate pipeline."""

    topic: str = Field(..., description="The debate topic")
    proponent_output: ArgumentOutput = Field(..., description="Proponent argument")
    opponent_output: RebuttalOutput = Field(..., description="Opponent rebuttal")
    critic_output: CriticOutput = Field(..., description="Position-neutral critic evaluation")
    judge_output: JudgeOutput = Field(..., description="Blind-judged rubric verdict")
    blind_mapping: dict[str, str] = Field(
        ..., description="Mapping of anonymized labels (Position A/B) to agent roles"
    )
    unblinded_winner: str = Field(
        ..., description="Final winner unblinded back to Proponent, Opponent, or Tie"
    )
    total_latency_seconds: float = Field(..., description="Total pipeline latency")
    total_tokens: int = Field(..., description="Total token consumption across all agent calls")
