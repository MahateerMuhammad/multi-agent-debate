"""Document ingestion, recursive chunking, and metadata preservation pipeline."""

from __future__ import annotations

import uuid

from app.retrieval.base import BaseVectorStore
from app.retrieval.schemas import Document, DocumentMetadata


def recursive_text_chunker(
    text: str,
    chunk_size: int = 384,
    chunk_overlap: int = 48,
) -> list[str]:
    """Recursively split text into semantic chunks breaking strictly on sentence boundaries."""
    clean_text = text.strip()
    if not clean_text:
        return []

    if len(clean_text) <= chunk_size:
        return [clean_text]

    chunks = []
    start = 0
    text_len = len(clean_text)

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(clean_text[start:])
            break

        # Try to break at paragraph, sentence, or clause boundary
        break_pos = clean_text.rfind("\n\n", start, end)
        if break_pos == -1 or break_pos < start + (chunk_size // 2):
            break_pos = clean_text.rfind(". ", start, end)
        if break_pos == -1 or break_pos < start + (chunk_size // 2):
            break_pos = clean_text.rfind("; ", start, end)
        if break_pos == -1 or break_pos < start + (chunk_size // 2):
            break_pos = end

        chunk_str = clean_text[start:break_pos].strip()
        if chunk_str:
            chunks.append(chunk_str)

        start = max(start + 1, break_pos - chunk_overlap)

    return [c for c in chunks if c]


class DocumentIngestor:
    """Pipeline for splitting raw documents into metadata-preserved vector store chunks."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        chunk_size: int = 384,
        chunk_overlap: int = 48,
    ):
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_raw_document(
        self,
        title: str,
        content: str,
        source: str = "unknown",
        author: str = "unknown",
        date: str = "unknown",
        doc_id: str | None = None,
    ) -> list[Document]:
        """Parse raw text, chunk with metadata preservation, and index in vector store."""
        base_id = doc_id or f"DOC-{uuid.uuid4().hex[:8]}"
        chunks = recursive_text_chunker(
            content, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

        documents = []
        for idx, chunk in enumerate(chunks, 1):
            meta = DocumentMetadata(
                title=title,
                source=source,
                author=author,
                date=date,
                doc_id=base_id,
            )
            doc = Document(
                id=f"{base_id}-chunk-{idx}",
                content=chunk,
                metadata=meta,
            )
            documents.append(doc)

        self.vector_store.add_documents(documents)
        return documents
