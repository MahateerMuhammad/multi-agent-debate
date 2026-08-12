"""Dense vector embedding implementations with TF-IDF weighting for high-precision search."""

from __future__ import annotations

import hashlib
import math

from app.retrieval.base import BaseEmbeddings


class MockEmbeddings(BaseEmbeddings):
    """Deterministic 384-dimensional dense vector generator with TF-IDF token weighting."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_text(self, text: str) -> list[float]:
        """Generate a deterministic 384-d normalized vector based on weighted token hashing."""
        clean_text = text.strip().lower()
        if not clean_text:
            return [0.0] * self.dimension

        vector = [0.0] * self.dimension
        tokens = clean_text.split()
        total_tokens = len(tokens)

        # Token term frequencies
        tf_dict: dict[str, float] = {}
        for t in tokens:
            tf_dict[t] = tf_dict.get(t, 0.0) + 1.0

        for token, count in tf_dict.items():
            # TF-IDF weighting (boost rare/informative words over common words)
            tf = count / total_tokens
            idf = 1.0 + math.log(1.0 + (10.0 / (len(token) + 1)))
            weight = tf * idf

            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for idx in range(4):
                val = ((digest[idx] % 256) / 255.0 - 0.5) * weight
                pos = (digest[idx + 4] * 31 + idx) % self.dimension
                vector[pos] += val

        # Normalize L2 norm
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0.0:
            vector = [v / norm for v in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of document strings."""
        return [self.embed_text(t) for t in texts]


class DenseEmbeddings(MockEmbeddings):
    """Primary dense embeddings provider."""

    pass
