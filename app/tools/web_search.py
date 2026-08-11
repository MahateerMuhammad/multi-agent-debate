"""Web search tool integration."""

class WebSearchTool:
    def __init__(self, api_key: str = ""):
        self.api_key = api_key

    def search(self, query: str) -> list[dict]:
        return []
