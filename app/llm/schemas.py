"""Pydantic data models and structured output schemas for LLM responses and debate agents."""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class LLMUsage(BaseModel):
    """Token usage metrics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel, Generic[T]):
    """Generic response wrapper for LLM calls containing payload, usage, and latency."""

    data: T
    raw_response: str = ""
    usage: LLMUsage = Field(default_factory=LLMUsage)
    latency_seconds: float = 0.0
    model_name: str = ""
    provider: str = ""


# Structured Output Schemas for Multi-Agent Debate


class ResearcherOutput(BaseModel):
    """Structured research findings for a given topic or claim."""

    topic: str = Field(..., description="The research topic or claim")
    summary: str = Field(..., description="High-level summary of findings")
    key_facts: list[str] = Field(default_factory=list, description="Key factual findings")
    sources: list[str] = Field(default_factory=list, description="Source documents or URLs")


class ArgumentOutput(BaseModel):
    """Structured proposition argument constructed by a Proponent agent."""

    claim: str = Field(..., description="The main claim being asserted")
    reasoning: list[str] = Field(..., description="Step-by-step reasoning points")
    supporting_evidence: list[str] = Field(
        default_factory=list, description="Evidence items supporting the claim"
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Self-assessed confidence score (0 to 1)"
    )


class RebuttalOutput(BaseModel):
    """Structured counter-argument constructed by an Opponent agent."""

    target_claim: str = Field(..., description="The claim being opposed")
    counter_arguments: list[str] = Field(..., description="Key rebuttal arguments")
    flaws_identified: list[str] = Field(
        default_factory=list, description="Flaws identified in opposing reasoning"
    )


class CritiqueOutput(BaseModel):
    """Structured evaluation of reasoning quality produced by a Critic agent."""

    argument_evaluated: str = Field(..., description="The argument that was evaluated")
    logical_fallacies: list[str] = Field(
        default_factory=list, description="Identified logical fallacies"
    )
    coherence_score: float = Field(..., ge=0.0, le=1.0, description="Coherence score (0 to 1)")
    suggestions: list[str] = Field(default_factory=list, description="Suggestions for improvement")


class EvidenceVerificationOutput(BaseModel):
    """Structured verification output produced by an Evidence verification agent."""

    claim: str = Field(..., description="The claim being verified")
    is_verified: bool = Field(..., description="True if verified by available evidence")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Verification confidence score")
    reasoning: str = Field(..., description="Justification for verification decision")


class VerdictOutput(BaseModel):
    """Structured final evaluation and winner determination produced by a Judge agent."""

    winner: str = Field(..., description="The winning side (e.g. Proponent or Opponent)")
    summary: str = Field(..., description="Summary justification for the final verdict")
    key_deciding_factors: list[str] = Field(
        ..., description="Deciding factors in the determination"
    )
    proponent_score: float = Field(
        ..., ge=0.0, le=1.0, description="Proponent score between 0 and 1"
    )
    opponent_score: float = Field(..., ge=0.0, le=1.0, description="Opponent score between 0 and 1")
