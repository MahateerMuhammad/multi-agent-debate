"""Agent output schemas and debate container data models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.llm.schemas import ArgumentOutput, RebuttalOutput


class SimpleDebateResult(BaseModel):
    """Container holding the results of a 2-agent Proponent vs Opponent debate."""

    topic: str = Field(..., description="The debate proposition topic")
    proponent_output: ArgumentOutput = Field(..., description="Proponent agent's argument")
    opponent_output: RebuttalOutput = Field(..., description="Opponent agent's rebuttal")
    total_latency_seconds: float = Field(..., description="Total elapsed latency for the debate")
    total_tokens: int = Field(..., description="Total token consumption across all calls")
