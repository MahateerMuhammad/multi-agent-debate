"""Evaluation Judge re-exports and quality auditing tools."""

from app.evaluation.benchmarks import (
    BENCHMARK_SUITE,
    BenchmarkReport,
    BenchmarkTestCase,
    run_benchmark_evaluations,
)
from app.evaluation.metrics import (
    compute_argument_coherence,
    compute_fallacy_density,
    compute_rebuttal_directness,
    compute_rubric_composite_score,
)

__all__ = [
    "compute_argument_coherence",
    "compute_fallacy_density",
    "compute_rebuttal_directness",
    "compute_rubric_composite_score",
    "BenchmarkTestCase",
    "BenchmarkReport",
    "BENCHMARK_SUITE",
    "run_benchmark_evaluations",
]
