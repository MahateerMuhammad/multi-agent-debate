"""Controlled Factorial Ablation Study and Candidate Pooling Matrix."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.dataset import RAGBenchmarkQuery
from app.evaluation.metrics import (
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_rag_f1_score,
    compute_recall_at_k,
)
from app.retrieval.bm25 import BM25OkapiIndexer
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import SemanticReranker
from app.retrieval.vectorstore import QdrantVectorStore


class AblationResult(BaseModel):
    """Evaluation output for a single pipeline configuration."""

    configuration_name: str
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    rag_f1_at_3: float
    mrr: float
    ndcg_at_3: float
    average_latency_ms: float


class FactorialAblationMatrix(BaseModel):
    """Clean Factorial Ablation isolating Reranker vs Fusion effects."""

    vector_only_mrr: float
    vector_plus_reranker_mrr: float
    hybrid_mrr: float
    hybrid_plus_reranker_mrr: float
    delta_reranker_isolated: float = Field(
        ..., description="MRR(Vector+Reranker) - MRR(Vector Only)"
    )
    delta_fusion_isolated: float = Field(..., description="MRR(Hybrid) - MRR(Vector Only)")
    delta_reranker_on_hybrid: float = Field(..., description="MRR(Hybrid+Reranker) - MRR(Hybrid)")


async def run_component_ablation_study(
    dataset: list[RAGBenchmarkQuery],
    vector_store: QdrantVectorStore,
    bm25_indexer: BM25OkapiIndexer,
) -> list[AblationResult]:
    """Execute clean 4-way Factorial Ablation Study isolating Reranker and Fusion effects."""
    eval_queries = [q for q in dataset if not q.is_negative]

    configurations: list[tuple[str, dict[str, Any]]] = [
        ("A. Vector Search Only", {"use_bm25": False, "use_reranker": False}),
        ("B. Vector + Semantic Reranker", {"use_bm25": False, "use_reranker": True}),
        ("C. Hybrid (BM25 + Vector / RRF)", {"use_bm25": True, "use_reranker": False}),
        ("D. Hybrid + Semantic Reranker", {"use_bm25": True, "use_reranker": True}),
    ]

    ablation_results: list[AblationResult] = []

    for name, config in configurations:
        p1_list, p3_list, p5_list = [], [], []
        r1_list, r3_list, r5_list = [], [], []
        mrr_list, ndcg_list, latencies = [], [], []

        use_bm25 = config["use_bm25"]
        use_reranker = config["use_reranker"]

        retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_indexer=bm25_indexer if use_bm25 else BM25OkapiIndexer(),
            reranker=SemanticReranker() if use_reranker else None,
            candidate_k=20,
            top_k=3,
            vector_weight=0.5 if use_bm25 else 1.0,
            bm25_weight=0.5 if use_bm25 else 0.0,
        )

        for q in eval_queries:
            t0 = time.perf_counter()

            if not use_bm25 and not use_reranker:
                # Vector Search Only
                vec_res = vector_store.search(query=q.query, top_k=3, score_threshold=0.0)
                results = vec_res
            elif not use_bm25 and use_reranker:
                # Vector + Reranker
                vec_candidates = vector_store.search(query=q.query, top_k=20, score_threshold=0.0)
                results = SemanticReranker().rerank(query=q.query, results=vec_candidates, top_k=3)
            else:
                # Hybrid / Hybrid + Reranker
                results = retriever.retrieve_evidence(query=q.query, top_k=3)

            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            p1_list.append(compute_precision_at_k(results, q.expected_doc_ids, k=1))
            p3_list.append(compute_precision_at_k(results, q.expected_doc_ids, k=3))
            p5_list.append(compute_precision_at_k(results, q.expected_doc_ids, k=5))

            r1_list.append(compute_recall_at_k(results, q.expected_doc_ids, k=1))
            r3_list.append(compute_recall_at_k(results, q.expected_doc_ids, k=3))
            r5_list.append(compute_recall_at_k(results, q.expected_doc_ids, k=5))

            mrr_list.append(compute_mrr(results, q.expected_doc_ids))
            ndcg_list.append(compute_ndcg_at_k(results, q.expected_doc_ids, k=3))

        avg_p1 = sum(p1_list) / len(p1_list)
        avg_p3 = sum(p3_list) / len(p3_list)
        avg_p5 = sum(p5_list) / len(p5_list)

        r1_clean = [x for x in r1_list if x is not None]
        r3_clean = [x for x in r3_list if x is not None]
        r5_clean = [x for x in r5_list if x is not None]

        avg_r1 = sum(r1_clean) / len(r1_clean) if r1_clean else 0.0
        avg_r3 = sum(r3_clean) / len(r3_clean) if r3_clean else 0.0
        avg_r5 = sum(r5_clean) / len(r5_clean) if r5_clean else 0.0

        avg_f1 = compute_rag_f1_score(avg_p3, avg_r3)
        avg_mrr = sum(mrr_list) / len(mrr_list)
        avg_ndcg = sum(ndcg_list) / len(ndcg_list)
        avg_lat = sum(latencies) / len(latencies)

        ablation_results.append(
            AblationResult(
                configuration_name=name,
                precision_at_1=round(avg_p1, 4),
                precision_at_3=round(avg_p3, 4),
                precision_at_5=round(avg_p5, 4),
                recall_at_1=round(avg_r1, 4),
                recall_at_3=round(avg_r3, 4),
                recall_at_5=round(avg_r5, 4),
                rag_f1_at_3=avg_f1,
                mrr=round(avg_mrr, 4),
                ndcg_at_3=round(avg_ndcg, 4),
                average_latency_ms=round(avg_lat, 2),
            )
        )

    return ablation_results


async def run_candidate_pooling_matrix(
    dataset: list[RAGBenchmarkQuery],
    vector_store: QdrantVectorStore,
    bm25_indexer: BM25OkapiIndexer,
) -> list[dict[str, Any]]:
    """Evaluate candidate pooling matrix across candidate_k and final_k values."""
    eval_queries = [q for q in dataset if not q.is_negative]

    candidate_k_options = [10, 20, 50]
    final_k_options = [1, 3, 5]

    matrix_results: list[dict[str, Any]] = []

    for cand_k in candidate_k_options:
        for fin_k in final_k_options:
            retriever = HybridRetriever(
                vector_store=vector_store,
                bm25_indexer=bm25_indexer,
                reranker=SemanticReranker(),
                candidate_k=cand_k,
                top_k=fin_k,
            )

            p_list, r_list, mrr_list, ndcg_list = [], [], [], []

            for q in eval_queries:
                results = retriever.retrieve_evidence(query=q.query, top_k=fin_k)

                p_list.append(compute_precision_at_k(results, q.expected_doc_ids, k=fin_k))
                r_val = compute_recall_at_k(results, q.expected_doc_ids, k=fin_k)
                if r_val is not None:
                    r_list.append(r_val)
                mrr_list.append(compute_mrr(results, q.expected_doc_ids))
                ndcg_list.append(compute_ndcg_at_k(results, q.expected_doc_ids, k=fin_k))

            avg_p = sum(p_list) / len(p_list)
            avg_r = sum(r_list) / len(r_list) if r_list else 0.0
            avg_f1 = compute_rag_f1_score(avg_p, avg_r)
            avg_mrr = sum(mrr_list) / len(mrr_list)
            avg_ndcg = sum(ndcg_list) / len(ndcg_list)

            matrix_results.append(
                {
                    "candidate_k": cand_k,
                    "final_k": fin_k,
                    "precision_at_k": round(avg_p, 4),
                    "recall_at_k": round(avg_r, 4),
                    "f1_at_k": avg_f1,
                    "mrr": round(avg_mrr, 4),
                    "ndcg_at_k": round(avg_ndcg, 4),
                }
            )

    return matrix_results
