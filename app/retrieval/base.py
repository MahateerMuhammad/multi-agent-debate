"""Abstract interfaces for embeddings generation and vendor-independent vector store providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.retrieval.schemas import Document, SearchResult


class BaseEmbeddings(ABC):
    """Abstract base class for vector embedding generators."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate a dense vector embedding for a single string query."""
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate dense vector embeddings for a list of document strings."""
        pass


class BaseVectorStore(ABC):
    """Abstract base class for vector databases (Qdrant, Pinecone, Chroma, etc.)."""

    @abstractmethod
    def add_documents(self, documents: list[Document]) -> list[str]:
        """Index a list of metadata-preserved documents into the vector store."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.70,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Perform similarity search returning top-k matching documents above threshold."""
        pass
