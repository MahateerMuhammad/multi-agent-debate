"""Automated benchmark dataset and evaluation suite for multi-agent debate systems."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.metrics import (
    compute_argument_coherence,
    compute_fallacy_density,
    compute_rag_f1_score,
)
from app.graph.workflow import build_debate_graph
from app.llm.base import BaseLLMProvider


class BenchmarkTestCase(BaseModel):
    """Single benchmark topic test case."""

    id: str = Field(..., description="Benchmark test ID")
    topic: str = Field(..., description="Debate proposition topic")
    category: str = Field(..., description="Domain category (ethics, AI, policy, economics)")
    expected_doc_ids: set[str] = Field(
        default_factory=set, description="Expected ground-truth document IDs for RAG recall"
    )


BENCHMARK_SUITE: list[BenchmarkTestCase] = [
    BenchmarkTestCase(
        id="BM-001",
        topic="Open-source LLMs pose lower systemic risk than proprietary API models",
        category="AI Safety & Policy",
        expected_doc_ids={"CSET-2026-01"},
    ),
    BenchmarkTestCase(
        id="BM-002",
        topic="Artificial Intelligence should replace human judges in traffic court",
        category="Legal Automation",
        expected_doc_ids={"NJAT-2025-04"},
    ),
    BenchmarkTestCase(
        id="BM-003",
        topic="Universal Basic Income should be implemented globally to address AI displacement",
        category="Economics & Social Policy",
        expected_doc_ids=set(),
    ),
]


class BenchmarkReport(BaseModel):
    """Aggregated benchmark evaluation summary report."""

    total_benchmarks_run: int
    average_latency_seconds: float
    average_tokens_consumed: float
    average_coherence_score: float
    average_fallacy_density: float
    average_context_precision: float
    average_context_recall: float
    average_rag_f1_score: float
    results: list[dict[str, Any]]


async def run_benchmark_evaluations(
    suite: list[BenchmarkTestCase] | None = None,
    llm_provider: BaseLLMProvider | None = None,
    vector_store: Any | None = None,
    max_rounds: int = 2,
) -> BenchmarkReport:
    """Run full evaluation suite across benchmark test cases and return aggregated report."""
    test_cases = suite or BENCHMARK_SUITE
    graph = build_debate_graph(llm_provider=llm_provider, vector_store=vector_store)

    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    tokens_list: list[int] = []
    coherences: list[float] = []
    fallacy_densities: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for test_case in test_cases:
        initial_state = {
            "topic": test_case.topic,
            "current_round": 0,
            "max_rounds": max_rounds,
            "confidence_threshold": 0.92,
            "improvement_threshold": 0.02,
            "proponent_history": [],
            "opponent_history": [],
            "evidence_history": [],
            "critic_history": [],
            "judge_history": [],
            "errors": [],
        }

        final_state = await graph.ainvoke(initial_state)

        # Compute metric evaluations
        prop_hist = final_state.get("proponent_history", [])
        crit_hist = final_state.get("critic_history", [])
        jdg_hist = final_state.get("judge_history", [])
        ev_hist = final_state.get("evidence_history", [])

        if prop_hist:
            latest_prop = prop_hist[-1]
            coherence = compute_argument_coherence(
                latest_prop.get("claim", ""), latest_prop.get("reasoning", [])
            )
        else:
            coherence = 0.0

        if crit_hist:
            latest_crit = crit_hist[-1]
            fallacies = latest_crit.get("logical_fallacies", [])
            f_density = compute_fallacy_density(fallacies, total_claims=2)
        else:
            f_density = 0.0

        comp_score_a, comp_score_b = 0.0, 0.0
        if jdg_hist:
            latest_jdg = jdg_hist[-1]
            comp_score_a = latest_jdg.get("total_score_a", 0.0)
            comp_score_b = latest_jdg.get("total_score_b", 0.0)

        precision = 1.0
        recall = 1.0
        f1 = 1.0
        if ev_hist and test_case.expected_doc_ids:
            sources = set(ev_hist[-1].get("sources_cited", []))
            matched = sources.intersection(test_case.expected_doc_ids)
            precision = len(matched) / max(len(sources), 1)
            recall = len(matched) / max(len(test_case.expected_doc_ids), 1)
            f1 = compute_rag_f1_score(precision, recall)

        lat = final_state.get("total_latency", 0.0)
        tok = final_state.get("total_tokens", 0)

        latencies.append(lat)
        tokens_list.append(tok)
        coherences.append(coherence)
        fallacy_densities.append(f_density)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        results.append(
            {
                "id": test_case.id,
                "topic": test_case.topic,
                "category": test_case.category,
                "rounds_run": final_state.get("current_round", 0),
                "winner": latest_jdg.get("winner", "Tie") if jdg_hist else "Unknown",
                "score_a": comp_score_a,
                "score_b": comp_score_b,
                "coherence": coherence,
                "fallacy_density": f_density,
                "context_precision": precision,
                "context_recall": recall,
                "rag_f1_score": f1,
                "latency_seconds": lat,
                "tokens": tok,
                "errors": final_state.get("errors", []),
            }
        )

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    avg_tok = sum(tokens_list) / len(tokens_list) if tokens_list else 0.0
    avg_coh = sum(coherences) / len(coherences) if coherences else 0.0
    avg_fall = sum(fallacy_densities) / len(fallacy_densities) if fallacy_densities else 0.0
    avg_prec = sum(precisions) / len(precisions) if precisions else 0.0
    avg_rec = sum(recalls) / len(recalls) if recalls else 0.0
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return BenchmarkReport(
        total_benchmarks_run=len(test_cases),
        average_latency_seconds=round(avg_lat, 2),
        average_tokens_consumed=round(avg_tok, 1),
        average_coherence_score=round(avg_coh, 2),
        average_fallacy_density=round(avg_fall, 2),
        average_context_precision=round(avg_prec, 2),
        average_context_recall=round(avg_rec, 2),
        average_rag_f1_score=round(avg_f1, 2),
        results=results,
    )
