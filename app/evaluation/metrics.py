"""Statistically rigorous Information Retrieval and RAG evaluation metrics.

Includes Wilson Score Intervals, Bootstrap CIs, de-duplicated nDCG@K, Confusion Matrix,
Correct Abstention Rate, Citation Entailment, and overall accuracy.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

from app.llm.schemas import RubricDimensionScore
from app.retrieval.schemas import SearchResult


def compute_argument_coherence(claim: str, reasoning_points: list[str]) -> float:
    """Compute structural coherence score based on claim clarity and reasoning point count."""
    if not claim or not claim.strip():
        return 0.0
    if not reasoning_points:
        return 0.2

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


def compute_precision_at_k(
    retrieved_docs: list[SearchResult],
    ground_truth_ids: set[str],
    k: int = 3,
) -> float:
    """Compute Precision@K = (unique relevant retrieved docs in top-K) / K."""
    if k <= 0 or not retrieved_docs:
        return 0.0

    top_k_docs = retrieved_docs[:k]
    seen_ids: set[str] = set()
    relevant_count = 0

    for res in top_k_docs:
        doc = res.document
        matched_id = None
        if doc.id in ground_truth_ids:
            matched_id = doc.id
        elif doc.metadata.doc_id in ground_truth_ids:
            matched_id = doc.metadata.doc_id

        if matched_id and matched_id not in seen_ids:
            seen_ids.add(matched_id)
            relevant_count += 1

    return round(relevant_count / k, 4)


def compute_recall_at_k(
    retrieved_docs: list[SearchResult],
    ground_truth_ids: set[str],
    k: int = 3,
) -> float | None:
    """Compute Recall@K = (unique relevant retrieved docs in top-K) / (total ground truth docs).

    Returns None for negative queries where ground_truth_ids is empty.
    """
    if not ground_truth_ids:
        return None
    if k <= 0 or not retrieved_docs:
        return 0.0

    top_k_docs = retrieved_docs[:k]
    retrieved_ids = set()
    for res in top_k_docs:
        retrieved_ids.add(res.document.id)
        retrieved_ids.add(res.document.metadata.doc_id)

    matched = retrieved_ids.intersection(ground_truth_ids)
    return round(len(matched) / len(ground_truth_ids), 4)


def compute_rag_f1_score(precision: float, recall: float | None) -> float:
    """Compute RAG F1 Score (harmonic mean of Context Precision and Context Recall)."""
    if recall is None or (precision + recall == 0.0):
        return 0.0
    return round(2 * (precision * recall) / (precision + recall), 4)


def compute_mrr(
    retrieved_docs: list[SearchResult],
    ground_truth_ids: set[str],
) -> float:
    """Compute Mean Reciprocal Rank (MRR) = 1 / rank of first relevant retrieved document."""
    if not ground_truth_ids or not retrieved_docs:
        return 0.0

    for rank, res in enumerate(retrieved_docs, 1):
        doc = res.document
        if doc.id in ground_truth_ids or doc.metadata.doc_id in ground_truth_ids:
            return round(1.0 / rank, 4)

    return 0.0


def compute_ndcg_at_k(
    retrieved_docs: list[SearchResult],
    ground_truth_ids: set[str],
    k: int = 3,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG@K) under document-level GT model.

    Duplicate hits matching an already-seen document ID contribute rel=0.0 to prevent DCG > IDCG.
    """
    if not ground_truth_ids or not retrieved_docs or k <= 0:
        return 0.0

    top_k_docs = retrieved_docs[:k]
    dcg = 0.0
    seen_gt: set[str] = set()

    for rank, res in enumerate(top_k_docs, 1):
        doc = res.document
        matched_id = None
        if doc.id in ground_truth_ids:
            matched_id = doc.id
        elif doc.metadata.doc_id in ground_truth_ids:
            matched_id = doc.metadata.doc_id

        if matched_id and matched_id not in seen_gt:
            seen_gt.add(matched_id)
            rel = 1.0
        else:
            rel = 0.0

        dcg += rel / math.log2(rank + 1)

    ideal_rel_count = min(len(ground_truth_ids), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_rel_count + 1))

    if idcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return round(min(max(ndcg, 0.0), 1.0), 4)


def compute_context_precision(
    retrieved_docs: list[SearchResult],
    relevant_doc_ids: set[str],
) -> float:
    """Alias for compute_precision_at_k using retrieved length."""
    k = len(retrieved_docs) if retrieved_docs else 1
    return compute_precision_at_k(retrieved_docs, relevant_doc_ids, k=k)


def compute_context_recall(
    retrieved_docs: list[SearchResult],
    expected_ground_truth_ids: set[str],
) -> float | None:
    """Alias for compute_recall_at_k using retrieved length."""
    k = len(retrieved_docs) if retrieved_docs else 1
    return compute_recall_at_k(retrieved_docs, expected_ground_truth_ids, k=k)


def compute_faithfulness_score(generated_claim: str, context_text: str) -> float:
    """Compute lexical faithfulness ratio of generated claim against retrieved context."""
    if not generated_claim or not context_text:
        return 0.0

    claim_words = set(re.findall(r"\b\w{4,}\b", generated_claim.lower()))
    if not claim_words:
        return 1.0

    context_words = set(re.findall(r"\b\w{4,}\b", context_text.lower()))
    grounded = claim_words.intersection(context_words)

    return round(len(grounded) / len(claim_words), 4)


def compute_abstention_accuracy(is_negative_query: bool, verification_status: str) -> float:
    """Compute abstention indicator for a single query."""
    if is_negative_query:
        return 1.0 if str(verification_status).lower() == "insufficient_evidence" else 0.0
    else:
        return 1.0 if str(verification_status).lower() != "insufficient_evidence" else 0.0


def compute_confusion_matrix_and_accuracy(
    eval_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute confusion matrices for negative-only subset and overall 50-query dataset.

    Statistically decouples negative specificity from 50-query Accuracy.
    """
    pos_records = [r for r in eval_records if not r.get("is_negative", False)]
    neg_records = [r for r in eval_records if r.get("is_negative", False)]

    # 1. Answerable (Positive) Subset Metrics (n=45)
    tp = sum(
        1
        for r in pos_records
        if str(r.get("verification_status", "")).lower() != "insufficient_evidence"
    )
    fn = len(pos_records) - tp

    # 2. Negative-Only Subset Metrics (n=5: TP and FN are 0 by construction)
    tn = sum(
        1
        for r in neg_records
        if str(r.get("verification_status", "")).lower() == "insufficient_evidence"
    )
    fp = len(neg_records) - tn

    correct_abstention_rate = (tn / len(neg_records)) if neg_records else 1.0
    false_answer_rate = (fp / len(neg_records)) if neg_records else 0.0

    total_queries = len(eval_records)
    overall_accuracy = ((tp + tn) / total_queries) if total_queries > 0 else 1.0

    return {
        "positive_subset": {"tp": tp, "fn": fn, "total_answerable": len(pos_records)},
        "negative_subset": {
            "tp": 0,
            "fn": 0,
            "tn": tn,
            "fp": fp,
            "total_negative": len(neg_records),
            "correct_abstention_rate": round(correct_abstention_rate, 4),
            "false_answer_rate": round(false_answer_rate, 4),
        },
        "overall_50_query_dataset": {
            "total_queries": total_queries,
            "correct_predictions": tp + tn,
            "overall_accuracy": round(overall_accuracy, 4),
        },
    }


def compute_citation_split_metrics(
    claims: list[str],
    citations: list[dict[str, Any]],
    retrieved_chunks: list[SearchResult],
) -> dict[str, float]:
    """Compute 4-way split citation metrics including Citation Entailment."""
    if not claims:
        return {
            "citation_metadata_accuracy": 1.0,
            "citation_completeness": 1.0,
            "citation_relatedness": 1.0,
            "citation_entailment": 1.0,
        }

    # 1. Metadata Accuracy
    valid_meta = sum(
        1 for cite in citations if cite.get("title") and cite.get("doc_id") and cite.get("source")
    )
    meta_acc = (valid_meta / len(citations)) if citations else 1.0

    # 2. Completeness
    completeness = min(len(citations), len(claims)) / len(claims)

    # 3. Relatedness
    retrieved_text = " ".join(r.document.content.lower() for r in retrieved_chunks)
    related_count = sum(
        1
        for cite in citations
        if str(cite.get("title", "")).lower() in retrieved_text
        or any(
            c.document.metadata.title.lower() in str(cite.get("title", "")).lower()
            for c in retrieved_chunks
        )
    )
    relatedness = (related_count / len(citations)) if citations else 1.0

    # 4. Citation Entailment (strict claim-to-passage lexical entailment)
    entailment_count = 0
    for claim in claims:
        claim_words = set(re.findall(r"\b\w{4,}\b", claim.lower()))
        if not claim_words:
            entailment_count += 1
            continue
        context_words = set(re.findall(r"\b\w{4,}\b", retrieved_text))
        if len(claim_words.intersection(context_words)) / len(claim_words) >= 0.70:
            entailment_count += 1
    entailment = (entailment_count / len(claims)) if claims else 1.0

    return {
        "citation_metadata_accuracy": round(meta_acc, 4),
        "citation_completeness": round(completeness, 4),
        "citation_relatedness": round(relatedness, 4),
        "citation_entailment": round(entailment, 4),
    }


def compute_wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute Wilson Score Interval for proportion metrics (Precision, Recall, Accuracy)."""
    if total <= 0:
        return 0.0, 0.0, 0.0

    p_hat = successes / total
    z = 1.96  # 95% confidence Z-score

    denominator = 1 + (z**2 / total)
    center = p_hat + (z**2 / (2 * total))
    adjusted_center = center / denominator

    spread = z * math.sqrt((p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2)))
    adjusted_spread = spread / denominator

    lower_ci = max(0.0, adjusted_center - adjusted_spread)
    upper_ci = min(1.0, adjusted_center + adjusted_spread)

    return round(p_hat, 4), round(lower_ci, 4), round(upper_ci, 4)


def compute_bootstrap_confidence_interval(
    values: list[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Compute 1,000-sample Bootstrap Confidence Interval for continuous IR metrics (MRR, nDCG)."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0

    n = len(values)
    mean_val = sum(values) / n

    if n <= 1:
        return round(mean_val, 4), 0.0, round(mean_val, 4), round(mean_val, 4)

    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    std_dev = math.sqrt(variance)

    rng = random.Random(seed)
    bootstrap_means: list[float] = []

    for _ in range(n_resamples):
        sample = [rng.choice(values) for _ in range(n)]
        bootstrap_means.append(sum(sample) / n)

    bootstrap_means.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_idx = int(alpha * n_resamples)
    upper_idx = int((1.0 - alpha) * n_resamples)

    lower_ci = max(0.0, bootstrap_means[lower_idx])
    upper_ci = min(1.0, bootstrap_means[upper_idx])

    return round(mean_val, 4), round(std_dev, 4), round(lower_ci, 4), round(upper_ci, 4)
