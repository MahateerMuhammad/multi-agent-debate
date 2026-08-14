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


def compute_reasoning_lexical_alignment(claim: str, reasoning_points: list[str]) -> float:
    """Compute lexical alignment score based on claim overlap and structural length."""
    if not claim or not claim.strip():
        return 0.0
    if not reasoning_points:
        return 0.20

    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "by",
        "about",
        "that",
        "this",
    }
    claim_words = set(
        w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", claim) if w.lower() not in stopwords
    )

    relevance_scores = []
    for pt in reasoning_points:
        pt_words = set(
            w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", pt) if w.lower() not in stopwords
        )
        if claim_words:
            overlap = len(claim_words.intersection(pt_words)) / len(claim_words)
        else:
            overlap = 0.5
        length_ratio = min(len(pt.split()) / 12.0, 1.0)
        relevance_scores.append(0.5 * overlap + 0.5 * length_ratio)

    avg_rel = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
    count_factor = min(len(reasoning_points) / 3.0, 1.0)

    return round(0.6 * avg_rel + 0.4 * count_factor, 4)


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
    """Compute token-level lexical faithfulness ratio against retrieved context."""

    if not generated_claim or not generated_claim.strip():
        return 0.0
    if not context_text or not context_text.strip():
        return 1.0

    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "by",
        "about",
        "that",
        "this",
        "these",
        "those",
        "from",
        "at",
        "as",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "not",
        "no",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "must",
        "shall",
        "will",
        "mock",
        "analysis",
        "item",
        "content",
        "regarding",
        "showing",
        "providing",
    }

    words = re.findall(r"\b[a-zA-Z0-9\-\.]{3,}\b", generated_claim.lower())
    claim_words = [w for w in words if w not in stopwords]

    if not claim_words:
        return 1.0

    context_words = set(re.findall(r"\b[a-zA-Z0-9\-\.]{3,}\b", context_text.lower()))
    grounded_count = sum(1 for w in claim_words if w in context_words)

    return round(grounded_count / len(claim_words), 4)


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
    """Compute 4-way split citation metrics including continuous Citation Entailment."""
    if not claims:
        return {
            "citation_metadata_accuracy": 1.0,
            "citation_completeness": 1.0,
            "citation_relatedness": 1.0,
            "citation_entailment": 1.0,
        }
    if not citations:
        return {
            "citation_metadata_accuracy": 0.0,
            "citation_completeness": 0.0,
            "citation_relatedness": 0.0,
            "citation_entailment": 0.0,
        }

    # 1. Metadata Accuracy
    valid_meta = sum(
        1 for cite in citations if cite.get("title") or cite.get("doc_id") or cite.get("source")
    )
    meta_acc = valid_meta / len(citations)

    # 2. Completeness
    completeness = min(len(citations) / max(len(claims), 1), 1.0)

    # 3. Relatedness (matching retrieved doc IDs, titles, sources, or content)
    retrieved_text = " ".join(r.document.content.lower() for r in retrieved_chunks)
    retrieved_doc_ids = set()
    retrieved_titles = set()
    for r in retrieved_chunks:
        if r.document.id:
            retrieved_doc_ids.add(str(r.document.id).lower())
        if r.document.metadata and r.document.metadata.doc_id:
            retrieved_doc_ids.add(str(r.document.metadata.doc_id).lower())
        if r.document.metadata and r.document.metadata.title:
            retrieved_titles.add(str(r.document.metadata.title).lower())

    related_count = 0
    for cite in citations:
        c_title = str(cite.get("title", "")).lower()
        c_doc_id = str(cite.get("doc_id", "")).lower()
        c_source = str(cite.get("source", "")).lower()

        if (
            c_doc_id in retrieved_doc_ids
            or c_title in retrieved_doc_ids
            or c_source in retrieved_doc_ids
            or any(t in c_title or c_title in t for t in retrieved_titles if t)
            or (c_title and c_title in retrieved_text)
            or (c_doc_id and c_doc_id in retrieved_text)
        ):
            related_count += 1

    relatedness = related_count / len(citations)

    # 4. Citation Entailment (lexical overlap ratio of claims against cited passage text)
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "by",
        "about",
        "that",
        "this",
        "these",
        "those",
        "from",
        "at",
        "as",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "not",
        "no",
        "can",
        "could",
    }
    all_claim_words: set[str] = set()

    for claim in claims:
        words = re.findall(r"\b[a-zA-Z0-9\-\.]{3,}\b", claim.lower())
        all_claim_words.update(w for w in words if w not in stopwords)

    if not all_claim_words:
        entailment = 1.0
    elif not retrieved_text:
        entailment = 0.0
    else:
        context_words = set(re.findall(r"\b[a-zA-Z0-9\-\.]{3,}\b", retrieved_text.lower()))
        grounded = all_claim_words.intersection(context_words)
        entailment = len(grounded) / len(all_claim_words)

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
    """Compute Wilson Score Interval for binomial proportion."""
    if total <= 0:
        return 0.0, 0.0, 0.0

    p_hat = successes / total
    z = 1.96  # 95% confidence level default

    denominator = 1.0 + (z**2 / total)
    center = p_hat + (z**2 / (2 * total))
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) + (z**2 / (4 * total))) / total)

    lower = max(0.0, (center - spread) / denominator)
    upper = min(1.0, (center + spread) / denominator)

    return round(p_hat, 4), round(lower, 4), round(upper, 4)


def compute_bootstrap_confidence_interval(
    values: list[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Compute non-parametric bootstrap confidence interval for mean of continuous metric."""
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


def estimate_llm_cost(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_rate_per_1k: float = 0.0015,
    completion_rate_per_1k: float = 0.0020,
) -> float:
    """Estimate total USD cost for LLM token usage based on per-1k-token pricing rates."""
    prompt_cost = (max(0, prompt_tokens) / 1000.0) * prompt_rate_per_1k
    completion_cost = (max(0, completion_tokens) / 1000.0) * completion_rate_per_1k
    return round(prompt_cost + completion_cost, 6)


def compute_completeness_score(text: str, key_aspects: list[str] | None = None) -> float:
    """Compute topic/aspect completeness score (0.0 to 1.0) based on analytical markers."""
    if not text or not text.strip():
        return 0.0

    clean_text = text.lower().strip()

    if key_aspects:
        found_count = 0
        for aspect in key_aspects:
            aspect_words = set(aspect.lower().split())
            if any(w in clean_text for w in aspect_words):
                found_count += 1
        return round(found_count / len(key_aspects), 4)

    # Distinct analytical marker categories
    categories = [
        ["claim", "proposition", "statement"],
        ["evidence", "citation", "source", "data", "patch"],
        ["rebuttal", "counter", "opponent", "however", "challenge"],
        ["risk", "uncertainty", "leakage", "limitation"],
        ["conclusion", "therefore", "verdict", "summary"],
    ]

    matched_categories = 0
    for cat in categories:
        if any(marker in clean_text for marker in cat):
            matched_categories += 1

    marker_ratio = matched_categories / len(categories)
    length_ratio = min(len(text.split()) / 80.0, 1.0)

    return round(0.6 * marker_ratio + 0.4 * length_ratio, 4)
