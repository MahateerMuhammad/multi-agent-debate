"""Pydantic data models and structured output schemas for LLM responses and debate agents."""

from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, Field, model_validator

T = TypeVar("T")


class LLMUsage(BaseModel):
    """Token usage metrics."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMResponse(BaseModel, Generic[T]):
    """Generic response wrapper for LLM calls containing payload, usage, and latency."""

    data: T
    raw_response: str = ""
    usage: LLMUsage = Field(default_factory=LLMUsage)
    latency_seconds: float = 0.0
    model_name: str = ""
    provider: str = ""


# Structured Output Schemas for Multi-Agent Debate

def robust_llm_validator(cls, values: Any) -> Any:
    """Shock absorber for open-source LLMs that slightly deviate from JSON schemas."""
    if not isinstance(values, dict):
        return values
        
    # Flatten common nested wrappers
    if len(values) == 1 and "evaluation" in values and isinstance(values["evaluation"], dict):
        values = values["evaluation"]
    if len(values) == 1 and "scores" in values and isinstance(values["scores"], dict):
        values = values["scores"]
        
    cleaned = {}
    key_mapping = {
        "central_claim": "claim",
        "argument_a": "argument_a_analysis",
        "argument_b": "argument_b_analysis",
    }
    
    for k, v in values.items():
        k = key_mapping.get(k, k)
        
        # Specific fix for JudgeOutput rubric_scores
        if k == "rubric_scores" and isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, str):
                    new_list.append({
                        "dimension": item, 
                        "score_a": 0.5, 
                        "score_b": 0.5, 
                        "justification": "Model failed to provide justification."
                    })
                elif isinstance(item, dict):
                    new_list.append(item)
            cleaned[k] = new_list
            continue

        # Flatten list of dicts to list of strings
        if isinstance(v, list):
            new_list = []
            for item in v:
                if isinstance(item, dict):
                    val = next(iter(item.values()), "")
                    new_list.append(str(val))
                else:
                    new_list.append(item)
            cleaned[k] = new_list
        else:
            cleaned[k] = v
            
    # Supply missing defaults for strict scalar fields to prevent catastrophic failure
    if cls.__name__ == "ArgumentOutput":
        cleaned.setdefault("claim", "Missing claim")
        cleaned.setdefault("reasoning", [])
    elif cls.__name__ == "CriticOutput":
        cleaned.setdefault("argument_a_analysis", "Missing analysis")
        cleaned.setdefault("argument_b_analysis", "Missing analysis")
        cleaned.setdefault("counterargument_quality_score", 0.5)
    elif cls.__name__ == "JudgeOutput":
        cleaned.setdefault("winner", "Tie")
        cleaned.setdefault("verdict_summary", "Missing verdict")
        cleaned.setdefault("rubric_scores", [])
        cleaned.setdefault("total_score_a", 0.5)
        cleaned.setdefault("total_score_b", 0.5)
        cleaned.setdefault("key_deciding_factors", [])
    elif cls.__name__ == "EvidenceVerificationOutput":
        cleaned.setdefault("claim", "Missing claim")
        cleaned.setdefault("status", "insufficient")
        cleaned.setdefault("is_verified", False)
        cleaned.setdefault("confidence", 0.5)
        cleaned.setdefault("reasoning", "Missing reasoning")
        
    return cleaned


class ResearcherOutput(BaseModel):
    """Structured research findings for a given topic or claim."""

    topic: str = Field(..., description="The research topic or claim")
    summary: str = Field(..., description="High-level summary of findings")
    key_facts: list[str] = Field(default_factory=list, description="Key factual findings")
    sources: list[str] = Field(default_factory=list, description="Source documents or URLs")


class ArgumentOutput(BaseModel):
    """Structured proposition argument constructed by a Proponent agent."""

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, values: Any) -> Any:
        return robust_llm_validator(cls, values)

    claim: str = Field(..., description="The main claim being asserted")
    reasoning: list[str] = Field(..., description="Step-by-step reasoning points")
    supporting_evidence: list[str] = Field(
        default_factory=list, description="Evidence items supporting the claim"
    )
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Self-assessed confidence score (0 to 1)"
    )


class RebuttalOutput(BaseModel):
    """Structured counter-argument constructed by an Opponent agent."""

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, values: Any) -> Any:
        return robust_llm_validator(cls, values)

    target_claim: str = Field(..., description="The claim being opposed")
    counter_arguments: list[str] = Field(..., description="Key rebuttal arguments")
    flaws_identified: list[str] = Field(
        default_factory=list, description="Flaws identified in opposing reasoning"
    )


class CriticOutput(BaseModel):
    """Structured position-neutral evaluation produced by a Critic agent."""

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, values: Any) -> Any:
        return robust_llm_validator(cls, values)

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

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, values: Any) -> Any:
        return robust_llm_validator(cls, values)

    claim: str = Field(..., description="The claim being verified")
    status: str = Field(
        ...,
        description="Status: supported | partially_supported | contradicted | insufficient",
    )
    is_verified: bool = Field(..., description="True if status is supported or partially_supported")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Verification confidence score")
    reasoning: str = Field(..., description="Justification for verification decision")
    sources_cited: list[str] = Field(
        default_factory=list, description="Preserved source titles, URLs, or DOIs"
    )


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

    @model_validator(mode="before")
    @classmethod
    def _validate(cls, values: Any) -> Any:
        return robust_llm_validator(cls, values)

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
