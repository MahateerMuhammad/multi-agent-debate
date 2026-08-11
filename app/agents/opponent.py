"""Opponent Agent: Formulates counter-arguments opposing the topic/proposition."""


class OpponentAgent:
    def __init__(self, name: str = "Opponent"):
        self.name = name

    def construct_counter_argument(self, topic: str, proponent_arg: str, evidence: list) -> str:
        return f"Opponent rebuttal to: {proponent_arg}"
