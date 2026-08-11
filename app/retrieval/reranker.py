"""Reranking module for retrieved context."""


class ContextReranker:
    def __init__(self, model_name: str = "cross-encoder"):
        self.model_name = model_name

    def rerank(self, query: str, docs: list[dict]) -> list[dict]:
        return docs
