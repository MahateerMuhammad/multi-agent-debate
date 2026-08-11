"""Retriever module for vector store and document retrieval."""

class DocumentRetriever:
    def __init__(self, vectorstore_path: str = "./data/vectorstore"):
        self.vectorstore_path = vectorstore_path

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return []
