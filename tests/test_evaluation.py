from __future__ import annotations

import pytest

from app.evaluation.benchmarks import run_benchmark_evaluations
from app.evaluation.metrics import (
    compute_argument_coherence,
    compute_fallacy_density,
    compute_rebuttal_directness,
    compute_rubric_composite_score,
)
from app.llm.providers.mock import MockLLMProvider
from app.llm.schemas import RubricDimensionScore


def test_compute_argument_coherence() -> None:
    """Test argument coherence scoring."""
    assert compute_argument_coherence("", []) == 0.0
    assert compute_argument_coherence("Claim A", []) == 0.2
    assert compute_argument_coherence("Claim A", ["Point 1"]) == 0.60
    assert compute_argument_coherence("Claim A", ["Point 1", "Point 2"]) == 0.80
    assert compute_argument_coherence("Claim A", ["Point 1", "Point 2", "Point 3"]) == 0.95


def test_compute_fallacy_density() -> None:
    """Test fallacy density calculation."""
    assert compute_fallacy_density([], total_claims=2) == 0.0
    assert compute_fallacy_density(["Strawman"], total_claims=2) == 0.5
    assert compute_fallacy_density(["Strawman", "Ad Hominem"], total_claims=2) == 1.0


def test_compute_rebuttal_directness() -> None:
    """Test rebuttal directness scoring."""
    assert compute_rebuttal_directness("", []) == 0.0
    target = "Open source AI models reduce systemic risk"
    counter_arguments = ["Open source AI models increase vulnerability"]
    score = compute_rebuttal_directness(target, counter_arguments)
    assert score >= 0.5


def test_compute_rubric_composite_score() -> None:
    """Test weighted composite score calculation across 5 dimensions."""
    rubric_scores = [
        RubricDimensionScore(
            dimension="correctness",
            score_a=1.0,
            score_b=0.5,
            justification="A is factually accurate",
        ),
        RubricDimensionScore(
            dimension="evidence_quality",
            score_a=1.0,
            score_b=0.5,
            justification="A cited evidence",
        ),
        RubricDimensionScore(
            dimension="reasoning",
            score_a=1.0,
            score_b=0.5,
            justification="A is coherent",
        ),
        RubricDimensionScore(
            dimension="relevance",
            score_a=1.0,
            score_b=0.5,
            justification="A is relevant",
        ),
        RubricDimensionScore(
            dimension="completeness",
            score_a=1.0,
            score_b=0.5,
            justification="A is complete",
        ),
    ]

    score_a, score_b = compute_rubric_composite_score(rubric_scores)
    assert score_a == 1.0
    assert score_b == 0.5


@pytest.mark.asyncio
async def test_run_benchmark_evaluations_mock() -> None:
    """Test automated benchmark suite execution using MockLLMProvider."""
    mock_llm = MockLLMProvider()
    report = await run_benchmark_evaluations(llm_provider=mock_llm, max_rounds=1)

    assert report.total_benchmarks_run == 3
    assert len(report.results) == 3
    assert report.average_coherence_score > 0.0
    assert report.results[0]["id"] == "BM-001"
