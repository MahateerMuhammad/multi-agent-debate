"""Evaluation metrics for debate argument quality, coherence, and rubric scoring."""

from __future__ import annotations

from app.llm.schemas import RubricDimensionScore


def compute_argument_coherence(claim: str, reasoning_points: list[str]) -> float:
    """Compute structural coherence score based on claim clarity and reasoning point count."""
    if not claim or not claim.strip():
        return 0.0
    if not reasoning_points:
        return 0.2

    # Higher score for structured, multi-point reasoning
    point_count = len(reasoning_points)
    if point_count >= 3:
        return 0.95
    elif point_count == 2:
        return 0.80
    return 0.60


def compute_fallacy_density(fallacies: list[str], total_claims: int = 1) -> float:
    """Compute logical fallacy density ratio (0.0 = zero fallacies, 1.0 = high fallacy rate)."""
    if total_claims <= 0:
        return 0.0
    fallacy_count = len(fallacies)
    density = fallacy_count / max(total_claims, 1)
    return min(density, 1.0)


def compute_rebuttal_directness(target_claim: str, counter_arguments: list[str]) -> float:
    """Evaluate directness and relevance of counterarguments against target claim."""
    if not target_claim or not counter_arguments:
        return 0.0

    target_words = set(target_claim.lower().split())
    overlap_counts = []
    for ca in counter_arguments:
        ca_words = set(ca.lower().split())
        overlap = len(target_words.intersection(ca_words))
        overlap_counts.append(overlap)

    avg_overlap = sum(overlap_counts) / len(counter_arguments) if counter_arguments else 0.0
    score = min(0.5 + (avg_overlap * 0.1), 1.0)
    return round(score, 2)


def compute_rubric_composite_score(
    rubric_scores: list[RubricDimensionScore],
    weights: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Compute weighted composite scores for Position A and Position B across rubric dimensions."""
    if not rubric_scores:
        return 0.0, 0.0

    default_weights = {
        "correctness": 0.25,
        "evidence_quality": 0.25,
        "reasoning": 0.25,
        "relevance": 0.15,
        "completeness": 0.10,
    }
    weight_map = weights or default_weights

    total_weight_a = 0.0
    total_weight_b = 0.0
    weight_sum = 0.0

    for score in rubric_scores:
        dim = score.dimension.lower()
        w = weight_map.get(dim, 0.20)
        total_weight_a += score.score_a * w
        total_weight_b += score.score_b * w
        weight_sum += w

    if weight_sum > 0:
        final_a = total_weight_a / weight_sum
        final_b = total_weight_b / weight_sum
    else:
        final_a = 0.0
        final_b = 0.0

    return round(final_a, 4), round(final_b, 4)
