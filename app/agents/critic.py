"""Critic Agent: Evaluates reasoning fallacies, logical consistency, and gaps in arguments."""

class CriticAgent:
    def __init__(self, name: str = "Critic"):
        self.name = name

    def critique(self, argument: str) -> dict:
        return {"argument": argument, "flaws": [], "strengths": []}
