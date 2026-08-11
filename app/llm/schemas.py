"""Pydantic data models and structured output schemas for LLM responses and debate agents."""

from __future__ import annotations

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


class CriticOutput(BaseModel):
    """Structured position-neutral evaluation produced by a Critic agent."""

    argument_a_analysis: str = Field(
        ..., description="Neutral analysis of Argument A strengths/weaknesses"
    )
    argument_b_analysis: str = Field(
        ..., description="Neutral analysis of Argument B strengths/weaknesses"
    )
    unsupported_claims: list[str] = Field(
        default_factory=list, description="Claims lacking supporting evidence"
    )
    logical_fallacies: list[str] = Field(
        default_factory=list, description="Identified logical fallacies"
    )
    missing_assumptions: list[str] = Field(
        default_factory=list, description="Unstated or questionable assumptions"
    )
    counterargument_quality_score: float = Field(
        ..., ge=0.0, le=1.0, description="Quality rating of counterarguments (0 to 1)"
    )
    contradictions_found: list[str] = Field(
        default_factory=list, description="Contradictions within or between arguments"
    )


class EvidenceVerificationOutput(BaseModel):
    """Structured verification output produced by an Evidence verification agent."""

    claim: str = Field(..., description="The claim being verified")
    is_verified: bool = Field(..., description="True if verified by available evidence")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Verification confidence score")
    reasoning: str = Field(..., description="Justification for verification decision")


class RubricDimensionScore(BaseModel):
    """Score and justification for a single rubric evaluation dimension."""

    dimension: str = Field(
        ...,
        description="Rubric dimension name (correctness, evidence_quality, reasoning, relevance)",
    )
    score_a: float = Field(..., ge=0.0, le=1.0, description="Score for Position A (0.0 to 1.0)")
    score_b: float = Field(..., ge=0.0, le=1.0, description="Score for Position B (0.0 to 1.0)")
    justification: str = Field(
        ..., description="Detailed explanation justifying the dimension score"
    )


class JudgeOutput(BaseModel):
    """Structured final evaluation and winner determination produced by a Judge agent."""

    winner: str = Field(..., description="Declared winner ('Position A', 'Position B', or 'Tie')")
    verdict_summary: str = Field(
        ..., description="Comprehensive explanation of final verdict decision"
    )
    rubric_scores: list[RubricDimensionScore] = Field(
        ..., description="Detailed 5-dimension rubric scores with explanations"
    )
    total_score_a: float = Field(..., ge=0.0, le=1.0, description="Overall score for Position A")
    total_score_b: float = Field(..., ge=0.0, le=1.0, description="Overall score for Position B")
    key_deciding_factors: list[str] = Field(
        ..., description="Primary deciding factors leading to the verdict"
    )
