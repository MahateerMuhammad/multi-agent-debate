"""Proponent Agent: Formulates arguments in favor of the topic/proposition."""

class ProponentAgent:
    def __init__(self, name: str = "Proponent"):
        self.name = name

    def construct_argument(self, topic: str, evidence: list) -> str:
        return f"Proponent argument for: {topic}"
