"""Cross-Encoder Reranker and Context Ordering engine for high-precision retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.retrieval.schemas import SearchResult


class BaseReranker(ABC):
    """Abstract interface for RAG document rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Rescore and rerank retrieved document chunks for query relevance."""
        pass


class SemanticReranker(BaseReranker):
    """Semantic Cross-Encoder Reranker with Lost-In-The-Middle context ordering."""

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 3,
    ) -> list[SearchResult]:
        """Rescore candidate chunks using exact term overlap and semantic score fusion."""
        if not results or not query:
            return []

        query_terms = set(query.lower().split())
        rescored_results: list[SearchResult] = []

        for res in results:
            doc_content = res.document.content.lower()
            doc_terms = set(doc_content.split())

            # 1. Term Overlap Ratio
            overlap = len(query_terms.intersection(doc_terms))
            overlap_ratio = overlap / max(len(query_terms), 1)

            # 2. Score Fusion (Vector Similarity + Reranker Term Score)
            fused_score = (res.score * 0.5) + (overlap_ratio * 0.5)
            fused_score = min(round(fused_score, 4), 1.0)

            rescored_results.append(SearchResult(document=res.document, score=fused_score))

        # Sort descending by fused relevance score
        rescored_results.sort(key=lambda r: r.score, reverse=True)
        top_results = rescored_results[:top_k]

        # Apply Lost-In-The-Middle Context Re-ordering
        return self._reorder_context(top_results)

    def _reorder_context(self, results: list[SearchResult]) -> list[SearchResult]:
        """Reorder context to mitigate LLM lost-in-the-middle attention degradation."""
        if len(results) <= 2:
            return results

        # Best item at top, 2nd best at bottom, rest in middle
        left = 0
        right = len(results) - 1

        sorted_items = list(results)
        res_arr: list[SearchResult | None] = [None] * len(sorted_items)

        for i, item in enumerate(sorted_items):
            if i % 2 == 0:
                res_arr[left] = item
                left += 1
            else:
                res_arr[right] = item
                right -= 1

        return [item for item in res_arr if item is not None]
