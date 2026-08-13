"""CLI Script for Phase 7 Evaluation and Experimental Framework.

Executes reproducible evaluation comparing Single-LLM, Two-Agent Debate,
Full Multi-Agent Debate (Fixed Rounds), and Full Multi-Agent Debate (Adaptive Stopping).
"""

from __future__ import annotations

import asyncio
from argparse import ArgumentParser
from collections.abc import Sequence
from typing import Any

from app.evaluation.benchmarks import BENCHMARK_SUITE
from app.evaluation.dataset import DEV_SET, FIFTY_QUERY_BENCHMARK, load_and_index_corpus
from app.evaluation.experiment_runner import EvaluationExperimentRunner
from app.llm.factory import get_llm_provider
from app.retrieval.bm25 import BM25OkapiIndexer
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vectorstore import QdrantVectorStore


async def main() -> None:
    parser = ArgumentParser(description="Phase 7 Multi-Agent Debate Evaluation Suite")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick evaluation on Benchmark Suite (3 representative queries)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run evaluation on 25-query Dev Set",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full evaluation on all 50 benchmark queries across 10 failure categories",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM Provider override ('mock', 'openrouter', 'qwen')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results",
        help="Directory path to save experiment results",
    )
    args = parser.parse_args()

    print("\n=======================================================")
    print("  PHASE 7: MULTI-AGENT DEBATE EVALUATION FRAMEWORK")
    print("=======================================================\n")

    provider = get_llm_provider(provider_name=args.provider)
    print(f"[+] Initialized LLM Provider: {provider.__class__.__name__}")

    print("[+] Building vector store and indexing ground-truth corpus...")
    vector_store = QdrantVectorStore(location=":memory:")
    chunks = load_and_index_corpus(vector_store)
    bm25_indexer = BM25OkapiIndexer()
    bm25_indexer.index_documents(chunks)

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_indexer=bm25_indexer,
        candidate_k=10,
        top_k=3,
    )
    print(f"[+] Corpus indexed: {len(chunks)} text chunks ready.")

    dataset: Sequence[Any]
    if args.quick:
        dataset = BENCHMARK_SUITE
        dataset_name = "Quick Benchmark Suite (3 Queries)"
    elif args.dev:
        dataset = DEV_SET
        dataset_name = "25-Query Dev Set"
    elif args.full:
        dataset = FIFTY_QUERY_BENCHMARK
        dataset_name = "Full 50-Query RAG Benchmark (10 Failure Categories)"
    else:
        dataset = BENCHMARK_SUITE
        dataset_name = "Quick Benchmark Suite (3 Queries)"

    print(f"[+] Selected Dataset: '{dataset_name}' ({len(dataset)} queries)")
    print("[+] Running 4 Experimental Conditions per query:")
    print("    1. Single-LLM Baseline")
    print("    2. Two-Agent Debate (Proponent vs Opponent)")
    print("    3. Full Multi-Agent Debate (Fixed Rounds)")
    print("    4. Full Multi-Agent Debate (Adaptive Stopping)")
    print("-" * 55)

    runner = EvaluationExperimentRunner(
        llm_provider=provider,
        hybrid_retriever=hybrid_retriever,
        vector_store=vector_store,
    )

    report = await runner.run_full_experiment_suite(dataset=dataset, dataset_name=dataset_name)

    json_path, md_path = runner.save_experiment_results(report, output_dir=args.output_dir)

    print("\n=======================================================")
    print("  EXPERIMENT COMPLETE - RESULTS SUMMARY")
    print("=======================================================")
    print(f"[+] Machine-Readable JSON: {json_path}")
    print(f"[+] Human-Readable Markdown: {md_path}\n")

    summary_md = runner.generate_markdown_summary(report)
    print(summary_md)
    print("\n=======================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
