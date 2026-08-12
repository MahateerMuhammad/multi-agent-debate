"""Qdrant Vector Database implementation preserving document metadata and similarity filtering."""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.retrieval.base import BaseEmbeddings, BaseVectorStore
from app.retrieval.embeddings import DenseEmbeddings
from app.retrieval.schemas import Document, DocumentMetadata, SearchResult


class QdrantVectorStore(BaseVectorStore):
    """Qdrant Vector Store providing metadata preservation and cosine similarity search."""

    def __init__(
        self,
        collection_name: str = "debate_documents",
        location: str = ":memory:",
        embeddings: BaseEmbeddings | None = None,
        dimension: int = 384,
    ):
        self.collection_name = collection_name
        self.embeddings = embeddings or DenseEmbeddings(dimension=dimension)
        self.dimension = dimension
        self.client = QdrantClient(location=location)

        # Re-create collection if it doesn't exist
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE),
            )

    def add_documents(self, documents: list[Document]) -> list[str]:
        """Index metadata-preserved documents into Qdrant vector database."""
        if not documents:
            return []

        points = []
        doc_ids = []

        texts = [doc.content for doc in documents]
        vectors = self.embeddings.embed_documents(texts)

        for doc, vector in zip(documents, vectors):  # noqa: B905
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.id))
            payload = {
                "content": doc.content,
                "doc_id": doc.metadata.doc_id,
                "title": doc.metadata.title,
                "source": doc.metadata.source,
                "author": doc.metadata.author,
                "date": doc.metadata.date,
                "chunk_id": doc.id,
            }
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            doc_ids.append(doc.id)

        self.client.upsert(collection_name=self.collection_name, points=points)
        return doc_ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.15,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Perform similarity search returning matching documents above score_threshold."""
        if not query or not query.strip():
            return []

        query_vector = self.embeddings.embed_text(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
        )

        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            meta = DocumentMetadata(
                title=payload.get("title", "unknown"),
                source=payload.get("source", "unknown"),
                author=payload.get("author", "unknown"),
                date=payload.get("date", "unknown"),
                doc_id=payload.get("doc_id", "unknown"),
            )
            doc = Document(
                id=payload.get("chunk_id", str(point.id)),
                content=payload.get("content", ""),
                metadata=meta,
            )
            results.append(SearchResult(document=doc, score=point.score))

        return results
