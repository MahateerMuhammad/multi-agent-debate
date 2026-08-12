"""Hybrid Search Retriever combining BM25 Lexical Search and Qdrant Vector Search."""

from __future__ import annotations

from typing import Any

from app.retrieval.base import BaseVectorStore
from app.retrieval.bm25 import BM25OkapiIndexer
from app.retrieval.reranker import BaseReranker, SemanticReranker
from app.retrieval.schemas import Document, SearchResult


class HybridRetriever:
    """Hybrid Retriever combining BM25 keyword matching and Dense Vector Search via RRF."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        bm25_indexer: BM25OkapiIndexer | None = None,
        reranker: BaseReranker | None = None,
        candidate_k: int = 20,
        top_k: int = 3,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ):
        self.vector_store = vector_store
        self.bm25_indexer = bm25_indexer or BM25OkapiIndexer()
        self.reranker = reranker or SemanticReranker()
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def index_documents(self, documents: list[Document]) -> None:
        """Index metadata-preserved document chunks into vector store and BM25 indexer."""
        self.vector_store.add_documents(documents)
        self.bm25_indexer.index_documents(documents)

    def retrieve_evidence(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float = 0.10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Execute hybrid search using Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking."""
        if not query or not query.strip():
            return []

        k = top_k if top_k is not None else self.top_k

        # 1. Fetch Candidate Lists from Vector Store & BM25 Indexer
        vector_candidates = self.vector_store.search(
            query=query,
            top_k=self.candidate_k,
            score_threshold=score_threshold,
            filter_metadata=filter_metadata,
        )
        bm25_candidates = self.bm25_indexer.search(
            query=query,
            top_k=self.candidate_k,
            filter_metadata=filter_metadata,
        )

        if not vector_candidates and not bm25_candidates:
            return []

        # 2. Compute Reciprocal Rank Fusion (RRF) Scores
        rrf_map: dict[str, float] = {}
        doc_map: dict[str, Document] = {}
        rrf_constant = 60.0

        for rank, res in enumerate(vector_candidates, 1):
            doc_id = res.document.id
            doc_map[doc_id] = res.document
            rrf_map[doc_id] = rrf_map.get(doc_id, 0.0) + self.vector_weight * (
                1.0 / (rrf_constant + rank)
            )

        for rank, res in enumerate(bm25_candidates, 1):
            doc_id = res.document.id
            doc_map[doc_id] = res.document
            rrf_map[doc_id] = rrf_map.get(doc_id, 0.0) + self.bm25_weight * (
                1.0 / (rrf_constant + rank)
            )

        # 3. Build Fused Candidate Pool
        fused_candidates: list[SearchResult] = []
        max_rrf = max(rrf_map.values()) if rrf_map else 1.0

        for doc_id, rrf_score in rrf_map.items():
            norm_score = min(round(rrf_score / max_rrf, 4), 1.0)
            fused_candidates.append(SearchResult(document=doc_map[doc_id], score=norm_score))

        fused_candidates.sort(key=lambda item: item.score, reverse=True)
        top_fused_candidates = fused_candidates[: self.candidate_k]

        # 4. Pass Top Fused Candidates to Semantic Cross-Encoder Reranker
        return self.reranker.rerank(query=query, results=top_fused_candidates, top_k=k)
