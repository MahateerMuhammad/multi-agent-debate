"""Schemas and data models for document ingestion, metadata preservation, and retrieval results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Preserved metadata associated with a document or chunk."""

    title: str = Field(..., description="Title of the source document")
    source: str = Field(default="unknown", description="Source URL, filename, or publication")
    author: str = Field(default="unknown", description="Document author or organization")
    date: str = Field(default="unknown", description="Publication date or date created")
    doc_id: str = Field(..., description="Unique document identifier")


class Document(BaseModel):
    """Document or chunk object indexed in the vector store."""

    id: str = Field(..., description="Unique chunk or document ID")
    content: str = Field(..., description="Text content of the document chunk")
    metadata: DocumentMetadata = Field(..., description="Associated document metadata")


class SearchResult(BaseModel):
    """Container for vector similarity search results."""

    document: Document = Field(..., description="The matched document chunk")
    score: float = Field(..., description="Similarity score (0.0 to 1.0)")
