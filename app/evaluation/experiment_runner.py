"""Experiment runner evaluating 4 debate configurations across 11 key metrics."""

from __future__ import annotations

import math
import platform
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.agents.opponent import OpponentAgent
from app.agents.proponent import ProponentAgent
from app.evaluation.metrics import (
    compute_citation_split_metrics,
    compute_completeness_score,
    compute_faithfulness_score,
    compute_fallacy_density,
    compute_reasoning_lexical_alignment,
    compute_rebuttal_directness,
    compute_recall_at_k,
    estimate_llm_cost,
)
from app.evaluation.schemas import (
    ConfidenceStatus,
    CorrectnessStatus,
    CostSource,
    EvaluationMode,
    UsageSource,
)
from app.graph.schemas import StopReason
from app.graph.workflow import build_debate_graph
from app.llm.base import BaseLLMProvider
from app.llm.factory import get_llm_provider
from app.llm.schemas import ArgumentOutput
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.schemas import SearchResult


def _stddev(vals: list[float]) -> float:
    """Calculate sample standard deviation."""
    if len(vals) <= 1:
        return 0.0
    mean_v = sum(vals) / len(vals)
    variance = sum((x - mean_v) ** 2 for x in vals) / (len(vals) - 1)
    return round(math.sqrt(variance), 4)


def get_system_metadata() -> dict[str, str]:
    """Capture environment and runtime reproducibility metadata."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": "unavailable",
    }


class MetricResult(BaseModel):
    """Container for key metrics measured for a single query execution."""

    correctness: Optional[float] = Field(default=None, description="Answer correctness (0.0 - 1.0)")  # noqa: UP045
    correctness_status: CorrectnessStatus = Field(
        ..., description="Evaluation status of correctness"
    )
    retrieval_recall: Optional[float] = Field(  # noqa: UP045
        default=None, description="Ground-truth doc recall (0.0 - 1.0)"
    )
    reasoning_lexical_alignment: float = Field(
        ..., description="Claim-to-reasoning lexical alignment & depth (0.0 - 1.0)"
    )
    evidence_grounding: float = Field(
        ..., description="Token-level lexical faithfulness ratio (0.0 - 1.0)"
    )
    citation_source_quality: float = Field(
        ..., description="Citation metadata & entailment accuracy (0.0 - 1.0)"
    )
    completeness: float = Field(
        ..., description="Analytical category coverage & length ratio (0.0 - 1.0)"
    )
    reported_confidence: Optional[float] = Field(
        default=None,
        description="Pure model/judge reported confidence rating (0.0 - 1.0), not calibrated.",
    )
    confidence_status: ConfidenceStatus = Field(..., description="Status of reported confidence")
    evaluator_confidence_score: float = Field(
        ..., description="Composite score of raw confidence and reasoning alignment."
    )
    latency_seconds: float = Field(..., description="Wall-clock time in seconds")
    usage_available: bool = Field(..., description="Indicates if token usage is available")
    prompt_tokens: Optional[int] = Field(default=None, description="Input/prompt token count")
    completion_tokens: Optional[int] = Field(default=None, description="Output/completion token count")
    total_tokens: Optional[int] = Field(default=None, description="Total prompt + completion tokens")
    usage_source: UsageSource = Field(..., description="Provenance of token usage")
    cost_source: CostSource = Field(..., description="Provenance of cost calculation")
    estimated_cost_usd: Optional[float] = Field(
        default=None, description="Total estimated cost in USD"
    )
    number_of_llm_calls: int = Field(..., description="Count of LLM API requests")
    number_of_debate_rounds: int = Field(..., description="Debate rounds executed")
    llm_call_traces: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-call trace ledger of tokens and usage source"
    )

    @property
    def reasoning_quality(self) -> float:
        """Alias for backward compatibility."""
        return self.reasoning_lexical_alignment

    @property
    def confidence(self) -> Optional[float]:
        """Alias for backward compatibility."""
        return self.reported_confidence


class QueryConditionResult(BaseModel):
    """Result for a single query under a specific experimental condition."""

    query_id: str
    query_text: str
    condition: str
    metrics: MetricResult
    output_summary: str
    retrieved_doc_ids: list[str]
    stop_reason: StopReason = Field(
        default=StopReason.max_rounds, description="Graph termination reason"
    )
    evaluation_mode: EvaluationMode = Field(
        default=EvaluationMode.synthetic, description="'synthetic' (mock) or 'real_model'"
    )
    round_traces: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-round state execution trajectory"
    )
    raw_details: dict[str, Any] = Field(default_factory=dict)


class ConditionAggregatedSummary(BaseModel):
    """Aggregated statistics for a single experimental condition across all benchmark queries."""

    condition_name: str
    sample_size: int

    # Means
    mean_correctness: Optional[float]  # noqa: UP045
    std_correctness: Optional[float]  # noqa: UP045
    mean_reasoning_quality: float
    mean_evidence_grounding: float
    mean_citation_source_quality: float
    mean_completeness: float
    mean_confidence: float
    mean_latency_seconds: float
    mean_total_tokens: float
    mean_estimated_cost_usd: float
    total_cost_usd: float
    mean_llm_calls: float
    mean_debate_rounds: float
    correctness_per_dollar: float = 0.0
    adaptive_cost_savings_pct: float = 0.0

    # Within-condition standard deviations
    std_reasoning_quality: float = 0.0
    std_evidence_grounding: float = 0.0
    std_citation_source_quality: float = 0.0
    std_completeness: float = 0.0
    std_confidence: float = 0.0
    std_latency_seconds: float = 0.0
    std_total_tokens: float = 0.0
    std_estimated_cost_usd: float = 0.0
    std_llm_calls: float = 0.0
    std_debate_rounds: float = 0.0


class FullExperimentReport(BaseModel):
    """Complete machine-readable experiment output."""

    timestamp: str
    evaluation_mode: EvaluationMode = Field(
        default=EvaluationMode.synthetic, description="'synthetic' (mock) or 'real_model'"
    )
    model_name: str
    dataset_name: str
    query_count: int
    system_metadata: dict[str, str] = Field(default_factory=dict)
    condition_summaries: dict[str, ConditionAggregatedSummary]
    detailed_query_results: list[QueryConditionResult]


class EvaluationExperimentRunner:
    """Orchestrates 4-way evaluation experiments isolating architecture and stopping effects."""

    def __init__(
        self,
        llm_provider: BaseLLMProvider | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        vector_store: Any | None = None,
    ):
        self.llm_provider = llm_provider or get_llm_provider()
        self.hybrid_retriever = hybrid_retriever
        self.vector_store = vector_store
        self.graph = build_debate_graph(
            llm_provider=self.llm_provider, vector_store=self.vector_store
        )
        self.is_mock = getattr(self.llm_provider, "provider_name", "").lower() == "mock"
        self.eval_mode = EvaluationMode.synthetic if self.is_mock else EvaluationMode.real_model

    async def _retrieve_context(self, query: str) -> tuple[list[SearchResult], str, list[str]]:
        """Retrieve fresh evidence passages if retriever is available and deduplicate doc IDs."""
        if not self.hybrid_retriever:
            return [], "", []

        results = list(self.hybrid_retriever.retrieve_evidence(query=query, top_k=3))
        raw_ids = [res.document.metadata.doc_id for res in results if res.document.metadata.doc_id]
        if not raw_ids:
            raw_ids = [res.document.id for res in results]

        retrieved_ids = list(dict.fromkeys([rid for rid in raw_ids if rid]))

        context_str = "\n\n".join(
            f"Doc ID: {res.document.metadata.doc_id or res.document.id}\n"
            f"Title: {res.document.metadata.title}\n"
            f"Content: {res.document.content}"
            for res in results
        )
        return results, str(context_str), list(retrieved_ids)

    async def run_single_llm(self, benchmark_query: Any) -> QueryConditionResult:
        """Condition 1: Single-LLM baseline direct execution (1 LLM call, 0 debate rounds)."""
        t0 = time.perf_counter()
        query_text = getattr(benchmark_query, "query", getattr(benchmark_query, "topic", ""))
        raw_expected: Any = getattr(benchmark_query, "expected_doc_ids", set())

        expected_ids: set[str] = (
            set(raw_expected) if isinstance(raw_expected, (set, list)) else set()
        )

        q_id = getattr(benchmark_query, "id", "Q-UNKNOWN")

        results, context_str, retrieved_ids = await self._retrieve_context(query_text)

        system_prompt = (
            "You are an expert analytical AI. Provide a well-reasoned, factual response "
            "to the user topic. Ground your analysis in provided context. Output JSON with fields: "
            "'claim', 'reasoning', 'confidence', 'citations'."
        )
        prompt = f"Topic: {query_text}\nContext:\n{context_str or 'No context available.'}"

        llm_resp = await self.llm_provider.generate_structured(
            prompt=prompt,
            response_model=ArgumentOutput,
            system_prompt=system_prompt,
        )
        latency = round(time.perf_counter() - t0, 3)

        data = llm_resp.data
        claim = getattr(data, "claim", str(data))
        reasoning = getattr(data, "reasoning", [])
        if isinstance(reasoning, str):
            reasoning = [reasoning]

        full_output_text = f"{claim} {' '.join(reasoning)}"
        reasoning_score = round(min(compute_reasoning_lexical_alignment(claim, reasoning), 0.70), 4)
        completeness = compute_completeness_score(full_output_text)
        grounding = compute_faithfulness_score(full_output_text, context_str)

        if expected_ids and retrieved_ids:
            retrieval_recall = compute_recall_at_k(results, expected_ids, k=len(results))
        else:
            retrieval_recall = None

        correctness = None
        correctness_status = CorrectnessStatus.not_evaluable

        citations = getattr(data, "supporting_evidence", [])
        if citations:
            cit_dicts = [
                {"title": str(c), "doc_id": str(c), "source": str(c)} if isinstance(c, str) else c
                for c in citations
            ]
            c_metrics = compute_citation_split_metrics([claim], cit_dicts, results)
            citation_quality = round(
                (
                    c_metrics["citation_metadata_accuracy"]
                    + c_metrics["citation_completeness"]
                    + c_metrics["citation_relatedness"]
                    + c_metrics["citation_entailment"]
                )
                / 4.0,
                4,
            )
        elif retrieved_ids:
            citation_quality = 0.0
        else:
            citation_quality = 1.0

        raw_conf = getattr(data, "confidence", None)
        reported_conf = float(raw_conf) if raw_conf is not None else None
        conf_status = (
            ConfidenceStatus.reported if reported_conf is not None else ConfidenceStatus.missing
        )
        evaluator_conf = round(0.70 + (reasoning_score * 0.20), 4)

        prompt_toks = llm_resp.usage.prompt_tokens
        comp_toks = llm_resp.usage.completion_tokens
        tok = llm_resp.usage.total_tokens

        usage_available = prompt_toks is not None
        if usage_available:
            prompt_cost = estimate_llm_cost(prompt_toks or 0, 0)
            comp_cost = estimate_llm_cost(0, comp_toks or 0)
            cost: Optional[float] = prompt_cost + comp_cost
        else:
            cost = None

        cost_source = (
            CostSource.local_pricing_estimate if usage_available else CostSource.unavailable
        )
        usage_source = (
            UsageSource.simulated
            if self.eval_mode == EvaluationMode.synthetic
            else (UsageSource.provider_reported if usage_available else UsageSource.unavailable)
        )

        metrics = MetricResult(
            correctness=correctness,
            correctness_status=correctness_status,
            retrieval_recall=retrieval_recall,
            reasoning_lexical_alignment=reasoning_score,
            evidence_grounding=round(grounding, 4),
            citation_source_quality=round(citation_quality, 4),
            completeness=completeness,
            reported_confidence=reported_conf,
            confidence_status=conf_status,
            evaluator_confidence_score=evaluator_conf,
            latency_seconds=latency,
            usage_available=usage_available,
            prompt_tokens=prompt_toks,
            completion_tokens=comp_toks,
            total_tokens=tok,
            usage_source=usage_source,
            cost_source=cost_source,
            estimated_cost_usd=round(cost, 6) if cost is not None else None,
            number_of_llm_calls=1,
            number_of_debate_rounds=0,
            llm_call_traces=[
                {
                    "node": "proponent",
                    "round": 0,
                    "prompt_tokens": prompt_toks,
                    "completion_tokens": comp_toks,
                    "total_tokens": tok,
                    "usage_source": usage_source.value,
                }
            ]
            if usage_available
            else [],
        )

        return QueryConditionResult(
            query_id=q_id,
            query_text=query_text,
            condition="single_llm",
            metrics=metrics,
            output_summary=f"Direct LLM Response: {claim}",
            retrieved_doc_ids=retrieved_ids,
            stop_reason=StopReason.direct_execution,
            evaluation_mode=self.eval_mode,
            round_traces=[],
            raw_details={
                "rounds_run": 0,
                "context_id": id(context_str) if context_str else "empty",
                "results_id": id(results),
            },
        )

    async def run_two_agent_debate(
        self, benchmark_query: Any, rounds: int = 2
    ) -> QueryConditionResult:
        """Condition 2: Two-Agent Debate (Proponent vs Opponent, 2 rounds)."""
        t0 = time.perf_counter()
        query_text = getattr(benchmark_query, "query", getattr(benchmark_query, "topic", ""))
        raw_expected: Any = getattr(benchmark_query, "expected_doc_ids", set())

        expected_ids: set[str] = (
            set(raw_expected) if isinstance(raw_expected, (set, list)) else set()
        )

        q_id = getattr(benchmark_query, "id", "Q-UNKNOWN")

        results, context_str, retrieved_ids = await self._retrieve_context(query_text)

        proponent = ProponentAgent(llm_provider=self.llm_provider)
        opponent = OpponentAgent(llm_provider=self.llm_provider)

        total_toks = 0
        p_toks = 0
        c_toks = 0
        prop_res = None
        opp_res = None
        round_traces = []
        llm_call_traces = []

        for r in range(rounds):
            round_topic = f"{query_text} [Debate Round {r + 1}]"
            prop_res = await proponent.construct_argument(round_topic)
            if prop_res.usage.prompt_tokens is not None:
                p_toks += prop_res.usage.prompt_tokens
                c_toks += prop_res.usage.completion_tokens or 0
                total_toks += prop_res.usage.total_tokens or (
                    prop_res.usage.prompt_tokens + (prop_res.usage.completion_tokens or 0)
                )

            opp_res = await opponent.construct_rebuttal(
                round_topic, proponent_argument=prop_res.data
            )
            if opp_res.usage.prompt_tokens is not None:
                p_toks += opp_res.usage.prompt_tokens
                c_toks += opp_res.usage.completion_tokens or 0
                total_toks += opp_res.usage.total_tokens or (
                    opp_res.usage.prompt_tokens + (opp_res.usage.completion_tokens or 0)
                )

            if prop_res.usage.prompt_tokens is not None:
                llm_call_traces.append(
                    {
                        "node": "proponent",
                        "round": r + 1,
                        "prompt_tokens": prop_res.usage.prompt_tokens,
                        "completion_tokens": prop_res.usage.completion_tokens,
                        "total_tokens": prop_res.usage.total_tokens,
                        "usage_source": prop_res.provider,
                    }
                )
            if opp_res.usage.prompt_tokens is not None:
                llm_call_traces.append(
                    {
                        "node": "opponent",
                        "round": r + 1,
                        "prompt_tokens": opp_res.usage.prompt_tokens,
                        "completion_tokens": opp_res.usage.completion_tokens,
                        "total_tokens": opp_res.usage.total_tokens,
                        "usage_source": opp_res.provider,
                    }
                )

            round_traces.append(
                {
                    "round": r + 1,
                    "proponent": {
                        "claim": prop_res.data.claim,
                        "reasoning_points": len(prop_res.data.reasoning),
                    },
                    "opponent": {
                        "target_claim": opp_res.data.target_claim,
                        "counter_arguments_count": len(opp_res.data.counter_arguments),
                    },
                }
            )

        latency = round(time.perf_counter() - t0, 3)

        last_prop = prop_res.data if prop_res else ArgumentOutput(claim="", reasoning=[])
        last_opp = (
            opp_res.data
            if opp_res
            else type(
                "DummyOpponent",
                (),
                {"counter_arguments": [], "target_claim": "", "flaws_identified": []},
            )()
        )

        claim = last_prop.claim
        reasoning = last_prop.reasoning
        counter_args = getattr(last_opp, "counter_arguments", [])

        coh = compute_reasoning_lexical_alignment(claim, reasoning)
        directness = compute_rebuttal_directness(claim, counter_args) if counter_args else 0.8
        reasoning_score = round(max(0.0, 0.5 * coh + 0.5 * directness), 4)

        full_output_text = f"{claim} {' '.join(reasoning)} {' '.join(counter_args)}"
        completeness = compute_completeness_score(full_output_text)
        grounding = compute_faithfulness_score(full_output_text, context_str)

        if expected_ids and retrieved_ids:
            retrieval_recall = compute_recall_at_k(results, expected_ids, k=len(results))
        else:
            retrieval_recall = None

        correctness = None
        correctness_status = CorrectnessStatus.not_evaluable

        citations = getattr(last_prop, "supporting_evidence", [])
        if citations:
            cit_dicts = [
                {"title": str(c), "doc_id": str(c), "source": str(c)} if isinstance(c, str) else c
                for c in citations
            ]
            c_metrics = compute_citation_split_metrics([claim], cit_dicts, results)
            citation_quality = round(
                (
                    c_metrics["citation_metadata_accuracy"]
                    + c_metrics["citation_completeness"]
                    + c_metrics["citation_relatedness"]
                    + c_metrics["citation_entailment"]
                )
                / 4.0,
                4,
            )
        elif retrieved_ids:
            citation_quality = 0.0
        else:
            citation_quality = 1.0

        raw_conf = getattr(last_prop, "confidence", None)
        reported_conf = float(raw_conf) if raw_conf is not None else None
        conf_status = (
            ConfidenceStatus.reported if reported_conf is not None else ConfidenceStatus.missing
        )
        evaluator_conf = round(0.60 + (reasoning_score * 0.25), 4)

        usage_available = p_toks > 0
        if usage_available:
            p_cost = estimate_llm_cost(p_toks, 0)
            c_cost = estimate_llm_cost(0, c_toks)
            cost: Optional[float] = p_cost + c_cost
        else:
            cost = None

        cost_source = (
            CostSource.local_pricing_estimate if usage_available else CostSource.unavailable
        )
        usage_source = (
            UsageSource.simulated
            if self.eval_mode == EvaluationMode.synthetic
            else (UsageSource.provider_reported if usage_available else UsageSource.unavailable)
        )

        metrics = MetricResult(
            correctness=correctness,
            correctness_status=correctness_status,
            retrieval_recall=retrieval_recall,
            reasoning_lexical_alignment=reasoning_score,
            evidence_grounding=round(grounding, 4),
            citation_source_quality=round(citation_quality, 4),
            completeness=completeness,
            reported_confidence=reported_conf,
            confidence_status=conf_status,
            evaluator_confidence_score=evaluator_conf,
            latency_seconds=latency,
            usage_available=usage_available,
            prompt_tokens=p_toks if usage_available else None,
            completion_tokens=c_toks if usage_available else None,
            total_tokens=total_toks if usage_available else None,
            usage_source=usage_source,
            cost_source=cost_source,
            estimated_cost_usd=round(cost, 6) if cost is not None else None,
            number_of_llm_calls=rounds * 2,
            number_of_debate_rounds=rounds,
            llm_call_traces=llm_call_traces,
        )

        return QueryConditionResult(
            query_id=q_id,
            query_text=query_text,
            condition="two_agent_debate",
            metrics=metrics,
            output_summary=f"Proponent Claim: {claim} | Opponent Rebuttals: {len(counter_args)}",
            retrieved_doc_ids=retrieved_ids,
            stop_reason=StopReason.max_rounds,
            evaluation_mode=self.eval_mode,
            round_traces=round_traces,
            raw_details={
                "rounds_run": rounds,
                "context_id": id(context_str) if context_str else "empty",
                "results_id": id(results),
            },
        )

    async def run_full_multi_agent(
        self, benchmark_query: Any, adaptive_stopping: bool = False, max_rounds: int = 3
    ) -> QueryConditionResult:
        """Condition 3 & 4: Full Multi-Agent Graph (Fixed vs Adaptive Stopping)."""
        t0 = time.perf_counter()
        query_text = getattr(benchmark_query, "query", getattr(benchmark_query, "topic", ""))
        raw_expected: Any = getattr(benchmark_query, "expected_doc_ids", set())

        expected_ids: set[str] = (
            set(raw_expected) if isinstance(raw_expected, (set, list)) else set()
        )

        q_id = getattr(benchmark_query, "id", "Q-UNKNOWN")

        results, context_str, retrieved_ids = await self._retrieve_context(query_text)

        conf_thresh = 0.90 if adaptive_stopping else 1.0
        imp_thresh = 0.03 if adaptive_stopping else 0.0

        initial_state: dict[str, Any] = {
            "topic": query_text,
            "current_round": 0,
            "max_rounds": max_rounds,
            "confidence_threshold": conf_thresh,
            "improvement_threshold": imp_thresh,
            "proponent_history": [],
            "opponent_history": [],
            "evidence_history": [],
            "critic_history": [],
            "judge_history": [],
            "errors": [],
        }

        final_state = await self.graph.ainvoke(initial_state)
        latency = round(time.perf_counter() - t0, 3)

        rounds_run = final_state.get("current_round", 1)
        prop_hist = final_state.get("proponent_history", [])
        crit_hist = final_state.get("critic_history", [])
        jdg_hist = final_state.get("judge_history", [])
        ev_hist = final_state.get("evidence_history", [])
        opp_hist = final_state.get("opponent_history", [])
        errors = final_state.get("errors", [])

        # Audit stop reason
        if any("failure" in str(e).lower() for e in errors):
            stop_reason = StopReason.fatal_system_error
        elif adaptive_stopping and len(jdg_hist) > 0:
            latest = jdg_hist[-1]
            max_s = max(latest.get("total_score_a", 0.0), latest.get("total_score_b", 0.0))
            if max_s >= conf_thresh:
                stop_reason = StopReason.confidence_threshold
            elif len(jdg_hist) >= 2:
                prev = jdg_hist[-2]
                prev_s = max(prev.get("total_score_a", 0.0), prev.get("total_score_b", 0.0))
                if (max_s - prev_s) < imp_thresh:
                    stop_reason = StopReason.quality_converged
                else:
                    stop_reason = StopReason.max_rounds
            else:
                stop_reason = StopReason.max_rounds
        else:
            stop_reason = StopReason.max_rounds

        round_traces = []
        for r in range(rounds_run):
            r_prop = prop_hist[r] if r < len(prop_hist) else {}
            r_opp = opp_hist[r] if r < len(opp_hist) else {}
            r_ev = ev_hist[r] if r < len(ev_hist) else {}
            r_crit = crit_hist[r] if r < len(crit_hist) else {}
            r_jdg = jdg_hist[r] if r < len(jdg_hist) else {}

            round_traces.append(
                {
                    "round": r + 1,
                    "proponent": {
                        "claim": r_prop.get("claim", ""),
                        "reasoning_points": len(r_prop.get("reasoning", [])),
                    },
                    "opponent": {
                        "target_claim": r_opp.get("target_claim", ""),
                        "counter_arguments_count": len(r_opp.get("counter_arguments", [])),
                    },
                    "evidence": {
                        "is_verified": r_ev.get("is_verified", True),
                        "sources_count": len(r_ev.get("sources_cited", [])),
                    },
                    "critic": {
                        "logical_fallacies": r_crit.get("logical_fallacies", []),
                        "flaws_identified": r_crit.get("flaws_identified", []),
                    },
                    "judge": {
                        "winner": r_jdg.get("winner", "Tie"),
                        "total_score_a": r_jdg.get("total_score_a", 0.0),
                        "verdict_summary": r_jdg.get("verdict_summary", ""),
                    },
                }
            )

        latest_prop = prop_hist[-1] if prop_hist else {}
        latest_opp = opp_hist[-1] if opp_hist else {}
        latest_crit = crit_hist[-1] if crit_hist else {}
        latest_jdg = jdg_hist[-1] if jdg_hist else {}

        claim = latest_prop.get("claim", "")
        reasoning = latest_prop.get("reasoning", [])
        fallacies = latest_crit.get("logical_fallacies", [])
        judge_rationale = latest_jdg.get("verdict_summary", "")

        coh = compute_reasoning_lexical_alignment(claim, reasoning)
        counter_args = latest_opp.get("counter_arguments", [])
        directness = compute_rebuttal_directness(claim, counter_args) if counter_args else 0.8
        f_density = compute_fallacy_density(fallacies)
        j_synth = compute_completeness_score(judge_rationale) if judge_rationale else 0.5

        reasoning_score = round(
            max(0.0, 0.35 * coh + 0.35 * directness + 0.30 * j_synth - (f_density * 0.1)),
            4,
        )

        full_output_text = (
            f"{claim} {' '.join(reasoning)} {' '.join(counter_args)} {judge_rationale}"
        )
        completeness = compute_completeness_score(full_output_text)
        grounding = compute_faithfulness_score(full_output_text, context_str)

        if expected_ids and retrieved_ids:
            retrieval_recall = compute_recall_at_k(results, expected_ids, k=len(results))
        else:
            retrieval_recall = None

        correctness = None
        correctness_status = CorrectnessStatus.not_evaluable

        sources_cited = ev_hist[-1].get("sources_cited", []) if ev_hist else []
        if not sources_cited:
            sources_cited = latest_prop.get("supporting_evidence", [])

        if sources_cited:
            cit_dicts = [
                {"title": str(c), "doc_id": str(c), "source": str(c)} if isinstance(c, str) else c
                for c in sources_cited
            ]
            c_metrics = compute_citation_split_metrics([claim], cit_dicts, results)
            citation_quality = round(
                (
                    c_metrics["citation_metadata_accuracy"]
                    + c_metrics["citation_completeness"]
                    + c_metrics["citation_relatedness"]
                    + c_metrics["citation_entailment"]
                )
                / 4.0,
                4,
            )
        elif retrieved_ids:
            citation_quality = 0.0
        else:
            citation_quality = 1.0

        raw_conf = latest_prop.get("confidence", None)
        reported_conf = float(raw_conf) if raw_conf is not None else None
        conf_status = (
            ConfidenceStatus.reported if reported_conf is not None else ConfidenceStatus.missing
        )
        jdg_conf = float(latest_jdg.get("total_score_a", 0.85))
        evaluator_conf = round(0.5 * jdg_conf + 0.5 * reasoning_score, 4)

        p_toks = final_state.get("prompt_tokens", 0)
        c_toks = final_state.get("completion_tokens", 0)
        tok = final_state.get("total_tokens", p_toks + c_toks)

        usage_available = p_toks > 0
        if usage_available:
            p_cost = estimate_llm_cost(p_toks, 0)
            c_cost = estimate_llm_cost(0, c_toks)
            cost: Optional[float] = p_cost + c_cost
        else:
            cost = None

        cost_source = (
            CostSource.local_pricing_estimate if usage_available else CostSource.unavailable
        )
        usage_source = (
            UsageSource.simulated
            if self.eval_mode == EvaluationMode.synthetic
            else (UsageSource.provider_reported if usage_available else UsageSource.unavailable)
        )

        llm_calls = rounds_run * 5
        cond_name = "full_multi_agent_adaptive" if adaptive_stopping else "full_multi_agent_fixed"

        llm_call_traces = final_state.get("llm_call_traces", [])

        metrics = MetricResult(
            correctness=correctness,
            correctness_status=correctness_status,
            retrieval_recall=retrieval_recall,
            reasoning_lexical_alignment=reasoning_score,
            evidence_grounding=round(grounding, 4),
            citation_source_quality=round(citation_quality, 4),
            completeness=completeness,
            reported_confidence=reported_conf,
            confidence_status=conf_status,
            evaluator_confidence_score=evaluator_conf,
            latency_seconds=latency,
            usage_available=usage_available,
            prompt_tokens=p_toks if usage_available else None,
            completion_tokens=c_toks if usage_available else None,
            total_tokens=tok if usage_available else None,
            usage_source=usage_source,
            cost_source=cost_source,
            estimated_cost_usd=round(cost, 6) if cost is not None else None,
            number_of_llm_calls=llm_calls,
            number_of_debate_rounds=rounds_run,
            llm_call_traces=llm_call_traces,
        )

        return QueryConditionResult(
            query_id=q_id,
            query_text=query_text,
            condition=cond_name,
            metrics=metrics,
            output_summary=(
                f"Winner: {latest_jdg.get('winner', 'Tie')} | Verdict: {judge_rationale[:80]}..."
            ),
            retrieved_doc_ids=retrieved_ids,
            stop_reason=stop_reason,
            evaluation_mode=self.eval_mode,
            round_traces=round_traces,
            raw_details={
                "stop_reason": stop_reason,
                "errors": errors,
                "context_id": id(context_str) if context_str else "empty",
                "results_id": id(results),
            },
        )

    async def run_full_experiment_suite(
        self, dataset: Sequence[Any], dataset_name: str = "Benchmark Suite"
    ) -> FullExperimentReport:
        """Run all 4 experimental conditions across all queries in the dataset."""
        detailed_results: list[QueryConditionResult] = []
        condition_buckets: dict[str, list[MetricResult]] = {
            "single_llm": [],
            "two_agent_debate": [],
            "full_multi_agent_fixed": [],
            "full_multi_agent_adaptive": [],
        }

        for q in dataset:
            res_1 = await self.run_single_llm(q)
            detailed_results.append(res_1)
            condition_buckets["single_llm"].append(res_1.metrics)

            res_2 = await self.run_two_agent_debate(q, rounds=2)
            detailed_results.append(res_2)
            condition_buckets["two_agent_debate"].append(res_2.metrics)

            res_3 = await self.run_full_multi_agent(q, adaptive_stopping=False, max_rounds=3)
            detailed_results.append(res_3)
            condition_buckets["full_multi_agent_fixed"].append(res_3.metrics)

            res_4 = await self.run_full_multi_agent(q, adaptive_stopping=True, max_rounds=5)
            detailed_results.append(res_4)
            condition_buckets["full_multi_agent_adaptive"].append(res_4.metrics)

            q_id_str = getattr(q, "id", "Q-UNKNOWN")
            res_1.raw_details["query_id"] = q_id_str
            res_2.raw_details["query_id"] = q_id_str
            res_3.raw_details["query_id"] = q_id_str
            res_4.raw_details["query_id"] = q_id_str

        condition_summaries: dict[str, ConditionAggregatedSummary] = {}
        fixed_cost = 0.0

        for cond_name, metrics_list in condition_buckets.items():
            n = len(metrics_list)
            if n == 0:
                continue

            correctness_vals = [m.correctness for m in metrics_list if m.correctness is not None]
            reasoning_vals = [m.reasoning_lexical_alignment for m in metrics_list]
            grounding_vals = [m.evidence_grounding for m in metrics_list]
            citation_vals = [m.citation_source_quality for m in metrics_list]
            completeness_vals = [m.completeness for m in metrics_list]
            confidence_vals = [
                m.reported_confidence for m in metrics_list if m.reported_confidence is not None
            ]
            latency_vals = [m.latency_seconds for m in metrics_list]
            token_vals = [float(m.total_tokens) for m in metrics_list if m.total_tokens is not None]
            cost_vals = [
                m.estimated_cost_usd for m in metrics_list if m.estimated_cost_usd is not None
            ]
            calls_vals = [float(m.number_of_llm_calls) for m in metrics_list]
            rounds_vals = [float(m.number_of_debate_rounds) for m in metrics_list]

            mean_corr = sum(correctness_vals) / len(correctness_vals) if correctness_vals else None
            mean_reas = sum(reasoning_vals) / n
            mean_grnd = sum(grounding_vals) / n
            mean_conf = sum(confidence_vals) / len(confidence_vals) if confidence_vals else 0.0
            mean_toks = sum(token_vals) / len(token_vals) if token_vals else 0.0
            mean_cost = sum(cost_vals) / len(cost_vals) if cost_vals else 0.0

            corr_per_dollar = (
                round(mean_corr / max(mean_cost, 0.000001), 2) if mean_corr is not None else 0.0
            )

            if cond_name == "full_multi_agent_fixed":
                fixed_cost = sum(cost_vals)

            savings_pct = 0.0
            if cond_name == "full_multi_agent_adaptive" and fixed_cost > 0.0:
                adap_cost = sum(cost_vals)
                savings_pct = round(((fixed_cost - adap_cost) / fixed_cost) * 100.0, 1)

            summary = ConditionAggregatedSummary(
                condition_name=cond_name,
                sample_size=n,
                mean_correctness=round(mean_corr, 4) if mean_corr is not None else None,
                mean_reasoning_quality=round(mean_reas, 4),
                mean_evidence_grounding=round(mean_grnd, 4),
                mean_citation_source_quality=sum(citation_vals) / n,
                mean_completeness=sum(completeness_vals) / n,
                mean_confidence=mean_conf,
                mean_latency_seconds=sum(latency_vals) / n,
                mean_total_tokens=mean_toks,
                mean_estimated_cost_usd=mean_cost,
                total_cost_usd=sum(cost_vals),
                mean_llm_calls=sum(calls_vals) / n,
                mean_debate_rounds=sum(rounds_vals) / n,
                correctness_per_dollar=corr_per_dollar,
                adaptive_cost_savings_pct=savings_pct,
                std_correctness=_stddev(correctness_vals) if len(correctness_vals) > 0 else None,
                std_reasoning_quality=_stddev(reasoning_vals),
                std_evidence_grounding=_stddev(grounding_vals),
                std_citation_source_quality=_stddev(citation_vals),
                std_completeness=_stddev(completeness_vals),
                std_confidence=_stddev(confidence_vals),
                std_latency_seconds=_stddev(latency_vals),
                std_total_tokens=_stddev(token_vals),
                std_estimated_cost_usd=_stddev(cost_vals),
                std_llm_calls=_stddev(calls_vals),
                std_debate_rounds=_stddev(rounds_vals),
            )
            condition_summaries[cond_name] = summary

        return FullExperimentReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            evaluation_mode=self.eval_mode,
            model_name=self.llm_provider.__class__.__name__,
            dataset_name=dataset_name,
            query_count=len(dataset),
            system_metadata=get_system_metadata(),
            condition_summaries=condition_summaries,
            detailed_query_results=detailed_results,
        )

    def save_experiment_results(
        self, report: FullExperimentReport, output_dir: str = "experiments/results"
    ) -> tuple[Path, Path]:
        """Save report to machine-readable JSON and human-readable Markdown summary."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        timestamp_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        json_file = out_path / f"evaluation_results_{timestamp_slug}.json"
        md_file = out_path / f"summary_{timestamp_slug}.md"

        with json_file.open("w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        md_content = self.generate_markdown_summary(report)
        with md_file.open("w", encoding="utf-8") as f:
            f.write(md_content)

        return json_file, md_file

    def generate_markdown_summary(self, report: FullExperimentReport) -> str:
        """Format an aggregated report into a structured Markdown document."""
        lines = ["# Phase 7 Evaluation Framework Report", ""]

        if report.evaluation_mode == EvaluationMode.synthetic:
            lines.extend(
                [
                    "> [!WARNING]",
                    "> **SYNTHETIC / INFRASTRUCTURE VALIDATION MODE**: Metrics were computed using "
                    f"`{report.model_name}`. Latency, token counts, and cost figures are synthetic "
                    "mock estimates designed for state transition, routing, and metric validation. "
                    "**They do not represent real-world LLM quality or API costs.**",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "> [!NOTE]",
                    "> **REAL MODEL EVALUATION MODE**: Metrics were computed using real LLM "
                    f"provider `{report.model_name}`.",
                    "",
                ]
            )

        py_v = report.system_metadata.get("python_version", "N/A")
        plat = report.system_metadata.get("platform", "N/A")

        lines.extend(
            [
                "> [!NOTE]",
                "> **Sample Size Caveat**: Within-condition standard deviations computed on small "
                "sample sizes are provisional estimates and must be re-verified on the full "
                "50-query benchmark suite before drawing final statistical conclusions.",
                "",
                f"- **Timestamp**: `{report.timestamp}`",
                f"- **Evaluation Mode**: `{report.evaluation_mode.value.upper()}`",
                f"- **Model**: `{report.model_name}`",
                f"- **Dataset**: `{report.dataset_name}` ({report.query_count} queries)",
                f"- **Python / System**: `{py_v}` on `{plat}`",
                "",
                "## Aggregated Means Across Experimental Conditions",
                "",
                "| Condition | Correctness | Reasoning Align | Grounding | Citation | Completeness "
                "| Confidence | Avg Latency | Avg Tokens | Avg Cost ($) | Avg Calls | Avg Rounds |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | "
                ":---: | :---: | :---: |",
            ]
        )

        for cond, s in report.condition_summaries.items():
            corr_str = f"{s.mean_correctness:.4f}" if s.mean_correctness is not None else "N/A"
            lines.append(
                f"| `{cond}` | {corr_str} | {s.mean_reasoning_quality:.4f} | "
                f"{s.mean_evidence_grounding:.4f} | {s.mean_citation_source_quality:.4f} | "
                f"{s.mean_completeness:.4f} | {s.mean_confidence:.4f} | "
                f"{s.mean_latency_seconds:.2f}s | {s.mean_total_tokens:.1f} | "
                f"${s.mean_estimated_cost_usd:.6f} | {s.mean_llm_calls:.1f} | "
                f"{s.mean_debate_rounds:.1f} |"
            )

        lines.extend(
            [
                "",
                "## Cost-Quality Efficiency & Resource Trade-Off Analysis",
                "",
                "| Condition | Mean Correctness | Reasoning Align | Total Cost ($) | "
                "Correctness / $ | Adaptive Savings |",
                "| :--- | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for cond, s in report.condition_summaries.items():
            savings_str = (
                f"{s.adaptive_cost_savings_pct:+.1f}%"
                if cond == "full_multi_agent_adaptive"
                else "Baseline"
            )
            corr_str = f"{s.mean_correctness:.4f}" if s.mean_correctness is not None else "N/A"
            lines.append(
                f"| `{cond}` | {corr_str} | {s.mean_reasoning_quality:.4f} | "
                f"${s.total_cost_usd:.6f} | {s.correctness_per_dollar:,.1f} | {savings_str} |"
            )

        lines.extend(
            [
                "",
                "## Within-Condition Standard Deviations (Across Queries)",
                "",
                "| Condition | σ(Correctness) | σ(Reasoning) | σ(Grounding) | σ(Citation) | "
                "σ(Completeness) | σ(Confidence) | σ(Latency) | σ(Tokens) |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for cond, s in report.condition_summaries.items():
            std_corr_str = f"{s.std_correctness:.4f}" if s.std_correctness is not None else "N/A"
            lines.append(
                f"| `{cond}` | {std_corr_str} | {s.std_reasoning_quality:.4f} | "
                f"{s.std_evidence_grounding:.4f} | {s.std_citation_source_quality:.4f} | "
                f"{s.std_completeness:.4f} | {s.std_confidence:.4f} | "
                f"{s.std_latency_seconds:.3f}s | {s.std_total_tokens:.1f} |"
            )

        lines.extend(
            [
                "",
                "## Detailed Per-Query Metric Breakdown",
                "",
                "| Query ID | Condition | Correctness | Reasoning | Grounding | Citation | "
                "Completeness | Confidence | Stop Reason | Latency | Tokens |",
                "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | "
                ":---: | :---: |",
            ]
        )
        for res in report.detailed_query_results:
            m = res.metrics
            corr_str = f"{m.correctness:.4f}" if m.correctness is not None else "N/A"
            conf_str = (
                f"{m.reported_confidence:.4f}" if m.reported_confidence is not None else "N/A"
            )
            tok_str = f"{m.total_tokens}" if m.total_tokens is not None else "N/A"
            lines.append(
                f"| `{res.query_id}` | `{res.condition}` | {corr_str} | "
                f"{m.reasoning_lexical_alignment:.4f} | {m.evidence_grounding:.4f} | "
                f"{m.citation_source_quality:.4f} | {m.completeness:.4f} | "
                f"{conf_str} | `{res.stop_reason.value}` | "
                f"{m.latency_seconds:.2f}s | {tok_str} |"
            )

        return "\n".join(lines)

        return "\n".join(lines)
