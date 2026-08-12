"""Okapi BM25 Lexical Keyword Search Indexer for exact term, acronym, and document ID retrieval."""

from __future__ import annotations

import math
import re
from typing import Any

from app.retrieval.schemas import Document, SearchResult


def default_tokenizer(text: str) -> list[str]:
    """Lowercase string and split into alphanumeric tokens preserving hyphens and acronyms."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-]+\b", text.lower())
    return [t for t in tokens if len(t) > 1]


class BM25OkapiIndexer:
    """Okapi BM25 Lexical Indexer providing score normalization and keyword search."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[Document] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self.total_docs: int = 0

    def index_documents(self, documents: list[Document]) -> None:
        """Index a list of metadata-preserved document chunks into the BM25 corpus."""
        if not documents:
            return

        self.documents = list(documents)
        self.total_docs = len(self.documents)
        self.doc_len = []
        self.doc_freqs = []

        df_counts: dict[str, int] = {}

        for doc in self.documents:
            tokens = default_tokenizer(doc.content)
            # Include metadata fields in term index for ID and author matching
            tokens.extend(default_tokenizer(doc.metadata.doc_id))
            tokens.extend(default_tokenizer(doc.metadata.title))

            self.doc_len.append(len(tokens))
            freqs: dict[str, int] = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1

            self.doc_freqs.append(freqs)

            for token in set(tokens):
                df_counts[token] = df_counts.get(token, 0) + 1

        self.avgdl = sum(self.doc_len) / self.total_docs if self.total_docs > 0 else 0.0

        # Calculate Okapi BM25 IDF for each term
        self.idf = {}
        for token, df in df_counts.items():
            # Standard Okapi BM25 IDF formula
            idf_val = math.log((self.total_docs - df + 0.5) / (df + 0.5) + 1.0)
            self.idf[token] = max(0.0, idf_val)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search indexed documents using Okapi BM25 scoring."""
        if not query or not self.documents or self.total_docs == 0:
            return []

        query_tokens = default_tokenizer(query)
        if not query_tokens:
            return []

        scores: list[float] = [0.0] * self.total_docs

        for idx, doc_freq in enumerate(self.doc_freqs):
            # Optional metadata filter check
            if filter_metadata:
                doc = self.documents[idx]
                meta_dict = doc.metadata.model_dump()
                if not all(meta_dict.get(k) == v for k, v in filter_metadata.items()):
                    continue

            doc_length = self.doc_len[idx]
            score = 0.0

            for token in query_tokens:
                if token not in doc_freq:
                    continue
                tf = doc_freq[token]
                idf = self.idf.get(token, 0.0)

                # Okapi BM25 TF component formula
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_length / self.avgdl))
                score += idf * (numerator / denominator)

            scores[idx] = score

        # Pair scores with documents and sort descending
        scored_pairs = [
            (self.documents[i], scores[i]) for i in range(self.total_docs) if scores[i] > 0.0
        ]
        scored_pairs.sort(key=lambda pair: pair[1], reverse=True)

        top_pairs = scored_pairs[:top_k]
        if not top_pairs:
            return []

        max_score = top_pairs[0][1] if top_pairs[0][1] > 0.0 else 1.0

        results: list[SearchResult] = []
        for doc, raw_score in top_pairs:
            # Normalize BM25 score to [0.0, 1.0] relative to max score in query result
            norm_score = min(round(raw_score / max_score, 4), 1.0)
            results.append(SearchResult(document=doc, score=norm_score))

        return results
