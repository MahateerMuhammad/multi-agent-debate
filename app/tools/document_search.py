"""Local document search tool integration."""

class DocumentSearchTool:
    def __init__(self, doc_dir: str = "./data/raw"):
        self.doc_dir = doc_dir

    def search(self, query: str) -> list[dict]:
        return []
