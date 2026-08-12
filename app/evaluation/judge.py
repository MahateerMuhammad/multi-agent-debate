"""Evaluation Judge re-exports and quality auditing tools."""

from app.evaluation.benchmarks import (
    BENCHMARK_SUITE,
    BenchmarkReport,
    BenchmarkTestCase,
    run_benchmark_evaluations,
)
from app.evaluation.metrics import (
    compute_abstention_accuracy,
    compute_argument_coherence,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness_score,
    compute_fallacy_density,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_rag_f1_score,
    compute_rebuttal_directness,
    compute_recall_at_k,
    compute_rubric_composite_score,
)

__all__ = [
    "compute_argument_coherence",
    "compute_fallacy_density",
    "compute_rebuttal_directness",
    "compute_rubric_composite_score",
    "compute_context_precision",
    "compute_context_recall",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_rag_f1_score",
    "compute_mrr",
    "compute_ndcg_at_k",
    "compute_faithfulness_score",
    "compute_abstention_accuracy",
    "BenchmarkTestCase",
    "BenchmarkReport",
    "BENCHMARK_SUITE",
    "run_benchmark_evaluations",
]
