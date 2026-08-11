"""Embedding generation utilities."""

class EmbeddingManager:
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name

    def embed_query(self, text: str) -> list[float]:
        return []
