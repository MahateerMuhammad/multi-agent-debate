"""Researcher Agent: Gathers background evidence and context for the debate topic."""

class ResearcherAgent:
    def __init__(self, name: str = "Researcher"):
        self.name = name

    def research(self, topic: str) -> dict:
        return {"topic": topic, "evidence": []}
