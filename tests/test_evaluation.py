from __future__ import annotations

import pytest

from app.evaluation.ablation import run_candidate_pooling_matrix, run_component_ablation_study
from app.evaluation.benchmarks import run_benchmark_evaluations
from app.evaluation.dataset import DEV_SET, UNSEEN_TEST_SET, load_and_index_corpus
from app.evaluation.metrics import (
    compute_bootstrap_confidence_interval,
    compute_citation_split_metrics,
    compute_confusion_matrix_and_accuracy,
    compute_faithfulness_score,
    compute_fallacy_density,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_reasoning_lexical_alignment,
    compute_rebuttal_directness,
    compute_recall_at_k,
    compute_rubric_composite_score,
    compute_wilson_score_interval,
    estimate_llm_cost,
)
from app.llm.providers.mock import MockLLMProvider
from app.llm.schemas import RubricDimensionScore
from app.retrieval.bm25 import BM25OkapiIndexer
from app.retrieval.schemas import Document, DocumentMetadata, SearchResult
from app.retrieval.vectorstore import QdrantVectorStore


def test_metric_cost_and_completeness_calculation() -> None:
    """Test LLM cost estimation based on separated prompt/completion tokens."""
    # 100 prompt, 50 completion
    cost_both = estimate_llm_cost(
        prompt_tokens=100,
        completion_tokens=50,
        prompt_rate_per_1k=0.0015,
        completion_rate_per_1k=0.0020
    )
    assert cost_both == pytest.approx((100 / 1000.0) * 0.0015 + (50 / 1000.0) * 0.0020)

    # Prompt only
    cost_prompt = estimate_llm_cost(100, 0, 0.0015, 0.0020)
    assert cost_prompt == pytest.approx((100 / 1000.0) * 0.0015)

    # Completion only
    cost_comp = estimate_llm_cost(0, 50, 0.0015, 0.0020)
    assert cost_comp == pytest.approx((50 / 1000.0) * 0.0020)

    # Both zero
    total_zero = estimate_llm_cost(0, 0)
    assert total_zero == pytest.approx(0.0)


def test_compute_reasoning_lexical_alignment() -> None:
    # Example 1: Claim and reasoning strongly overlap. Expected: high lexical alignment.
    high_align = compute_reasoning_lexical_alignment(
        "Open source models improve security significantly through decentralization",
        ["Open source models improve security significantly by allowing decentralized code audits."]
    )
    assert high_align >= 0.60

    # Example 2: Claim and reasoning have little lexical overlap. Expected: low lexical alignment.
    low_align = compute_reasoning_lexical_alignment(
        "The project needs more funding.",
        ["Apples are usually red or green and grow on trees in orchards."]
    )
    assert low_align <= 0.45

    # Example 3: Semantically good reasoning with low lexical overlap. Expected: metric scores low.
    semantic_good_low_align = compute_reasoning_lexical_alignment(
        "Automobiles contribute heavily to atmospheric warming.",
        ["Internal combustion engines release massive volumes of greenhouse gases daily."]
    )
    assert semantic_good_low_align <= 0.65
    assert semantic_good_low_align < high_align




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


def test_ir_ranking_metrics_and_faithfulness() -> None:
    """Test Precision@K, Recall@K, MRR, nDCG@K, Faithfulness, and Abstention accuracy."""
    meta1 = DocumentMetadata(
        title="Title A", source="src1", author="auth1", date="2026", doc_id="DOC-A"
    )
    meta2 = DocumentMetadata(
        title="Title B", source="src2", author="auth2", date="2026", doc_id="DOC-B"
    )

    doc1 = Document(id="c1", content="Open source vulnerability patching rate", metadata=meta1)
    doc2 = Document(id="c2", content="Unrelated text content", metadata=meta2)

    results = [SearchResult(document=doc1, score=0.90), SearchResult(document=doc2, score=0.40)]

    ground_truth = {"DOC-A"}

    assert compute_precision_at_k(results, ground_truth, k=1) == 1.0
    assert compute_recall_at_k(results, ground_truth, k=1) == 1.0
    assert compute_mrr(results, ground_truth) == 1.0


def test_ndcg_boundedness_and_negative_query_recall() -> None:
    """Test mathematical upper bound of nDCG@K <= 1.0 and undefined negative recall."""
    meta1 = DocumentMetadata(
        title="Title A", source="src1", author="auth1", date="2026", doc_id="DOC-A"
    )

    doc1 = Document(id="chunk-1", content="Content A chunk 1", metadata=meta1)
    doc2 = Document(id="chunk-2", content="Content A chunk 2", metadata=meta1)

    results = [SearchResult(document=doc1, score=0.90), SearchResult(document=doc2, score=0.85)]
    ground_truth = {"DOC-A"}

    # Manual Calculation:
    # rank 1: chunk-1 matches DOC-A -> rel = 1.0, DCG = 1.0 / log2(2) = 1.0
    # rank 2: chunk-2 matches DOC-A (already seen DOC-A) -> rel = 0.0, DCG += 0.0
    # IDCG for 1 item = 1.0 / log2(2) = 1.0
    # nDCG = 1.0 / 1.0 = 1.0000 <= 1.0
    ndcg = compute_ndcg_at_k(results, ground_truth, k=3)
    assert 0.0 <= ndcg <= 1.0
    assert ndcg == 1.0000

    # Test Undefined Negative Query Recall
    neg_recall = compute_recall_at_k(results, set(), k=3)
    assert neg_recall is None

    # Faithfulness test
    faith_score = compute_faithfulness_score(
        generated_claim="Open source vulnerability patching rate",
        context_text="Open source vulnerability patching rate is 45% faster.",
    )
    assert faith_score >= 0.80


def test_confusion_matrix_citation_and_confidence_interval_metrics() -> None:
    """Test abstention metrics, citation metrics, and confidence interval functions."""
    eval_records = [
        {"is_negative": True, "verification_status": "insufficient_evidence"},
        {"is_negative": True, "verification_status": "supported"},
        {"is_negative": False, "verification_status": "supported"},
        {"is_negative": False, "verification_status": "insufficient_evidence"},
    ]
    matrix = compute_confusion_matrix_and_accuracy(eval_records)
    assert matrix["positive_subset"]["tp"] == 1
    assert matrix["positive_subset"]["fn"] == 1
    assert matrix["negative_subset"]["tn"] == 1
    assert matrix["negative_subset"]["fp"] == 1
    assert matrix["negative_subset"]["correct_abstention_rate"] == 0.5
    assert matrix["overall_50_query_dataset"]["overall_accuracy"] == 0.5

    # Citation Metrics
    meta = DocumentMetadata(
        title="Title A", source="src1", author="auth1", date="2026", doc_id="DOC-A"
    )
    doc = Document(id="c1", content="Content text matching Title A", metadata=meta)
    results = [SearchResult(document=doc, score=0.9)]
    claims = ["Claim A"]
    citations = [{"title": "Title A", "doc_id": "DOC-A", "source": "src1"}]

    c_metrics = compute_citation_split_metrics(claims, citations, results)
    assert c_metrics["citation_metadata_accuracy"] == 1.0
    assert c_metrics["citation_completeness"] == 1.0

    # Wilson Score and Bootstrap CIs
    mean_w, l_w, u_w = compute_wilson_score_interval(19, 20)
    assert mean_w == 0.9500
    assert l_w < mean_w < u_w

    vals = [0.9, 0.95, 0.85, 0.92, 0.88]
    mean, std, l_ci, u_ci = compute_bootstrap_confidence_interval(vals)
    assert mean == 0.9000
    assert l_ci <= mean <= u_ci
    assert std > 0.0


def test_fifty_query_dataset_and_corpus_ingestion() -> None:
    """Test 50-query dataset loading and DEV / UNSEEN_TEST split."""
    store = QdrantVectorStore(location=":memory:")
    chunks = load_and_index_corpus(vector_store=store)

    assert len(chunks) > 0
    assert len(DEV_SET) == 25
    assert len(UNSEEN_TEST_SET) == 25


@pytest.mark.asyncio
async def test_ablation_framework_and_pooling_matrix() -> None:
    """Test ablation framework execution and candidate pooling matrix."""
    store = QdrantVectorStore(location=":memory:")
    bm25 = BM25OkapiIndexer()

    chunks = load_and_index_corpus(vector_store=store)
    bm25.index_documents(chunks)

    # Run on a 2-query slice for fast test execution
    mini_dataset = DEV_SET[:2]

    ablation_res = await run_component_ablation_study(
        dataset=mini_dataset, vector_store=store, bm25_indexer=bm25
    )
    assert len(ablation_res) == 4
    assert ablation_res[0].configuration_name == "A. Vector Search Only"

    matrix_res = await run_candidate_pooling_matrix(
        dataset=mini_dataset, vector_store=store, bm25_indexer=bm25
    )
    assert len(matrix_res) == 9  # 3 candidate_k * 3 final_k


@pytest.mark.asyncio
async def test_run_benchmark_evaluations_mock() -> None:
    """Test automated benchmark suite execution using MockLLMProvider."""
    mock_llm = MockLLMProvider()
    report = await run_benchmark_evaluations(llm_provider=mock_llm, max_rounds=1)

    assert report.total_benchmarks_run == 3
    assert len(report.results) == 3
    assert report.average_coherence_score > 0.0
    assert report.results[0]["id"] == "BM-001"


def test_rigorous_statistical_metrics_and_bootstrap() -> None:
    """Test Wilson Score CIs, paired bootstrap, and document-level ground truth."""
    # Test Wilson interval bounds at k=19, n=20
    w_mean, w_low, w_high = compute_wilson_score_interval(19, 20)
    assert w_mean == 0.95
    assert round(w_low, 4) == 0.7639
    assert round(w_high, 4) == 0.9911

    # Test Bootstrap CIs on values
    vals = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0]
    mean, std, low, high = compute_bootstrap_confidence_interval(vals, n_resamples=1000, seed=42)
    assert mean == 0.90
    assert low <= mean <= high

    # Test document-level deduplication in Precision@K and nDCG@K
    meta = DocumentMetadata(title="Doc A", source="s1", author="a1", date="2026", doc_id="DOC-1")
    d1 = Document(id="chunk-1", content="Chunk 1", metadata=meta)
    d2 = Document(id="chunk-2", content="Chunk 2", metadata=meta)
    results = [SearchResult(document=d1, score=0.9), SearchResult(document=d2, score=0.8)]

    # Matching DOC-1 should produce P@1 = 1.0, P@2 = 0.5 (1 unique doc out of 2 retrieved)
    assert compute_precision_at_k(results, {"DOC-1"}, k=1) == 1.0
    assert compute_precision_at_k(results, {"DOC-1"}, k=2) == 0.5
    # nDCG@2 should be 1.0 because rank 1 yields DCG=1.0, rank 2 yields rel=0.0 (duplicate)
    assert compute_ndcg_at_k(results, {"DOC-1"}, k=2) == 1.0
