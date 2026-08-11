"""Critic Agent: Position-neutral analyst evaluating argument validity, fallacies, and flaws."""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.llm.base import BaseLLMProvider
from app.llm.schemas import ArgumentOutput, CriticOutput, LLMResponse, RebuttalOutput


class CriticAgent(BaseAgent):
    """Critic Agent performs a strictly position-neutral evaluation of debate arguments."""

    def __init__(
        self,
        name: str = "Critic",
        llm_provider: BaseLLMProvider | None = None,
    ):
        super().__init__(
            name=name,
            role="Position-Neutral Analytical Critic",
            description=(
                "Evaluates Proponent and Opponent arguments objectively for logical fallacies, "
                "unsupported claims, missing assumptions, and structural flaws."
            ),
            llm_provider=llm_provider,
        )

    @property
    def agent_type(self) -> str:
        return "critic"

    async def evaluate_debate(
        self,
        topic: str,
        proponent_argument: ArgumentOutput,
        opponent_rebuttal: RebuttalOutput,
        context: dict[str, Any] | None = None,
    ) -> LLMResponse[CriticOutput]:
        """Perform a position-neutral analysis of both Proponent and Opponent positions."""
        clean_topic = self.validate_input(topic)

        system_prompt = self._format_system_prompt(
            "You are a completely position-neutral, objective critical analyst.\n"
            "Evaluate Argument A (Proponent) and Argument B (Opponent) with strict impartiality.\n"
            "Identify:\n"
            "1. Unsupported claims (statements lacking empirical backing)\n"
            "2. Logical fallacies (ad hominem, strawman, false dilemma, etc.)\n"
            "3. Missing or questionable implicit assumptions\n"
            "4. Quality rating of counterarguments (score 0.0 to 1.0)\n"
            "5. Internal or cross-argument contradictions."
        )

        user_prompt = (
            f"Debate Topic: {clean_topic}\n\n"
            f"=== ARGUMENT A (Proponent Case) ===\n"
            f"Claim: {proponent_argument.claim}\n"
            f"Reasoning: {', '.join(proponent_argument.reasoning)}\n"
            f"Evidence: {', '.join(proponent_argument.supporting_evidence)}\n\n"
            f"=== ARGUMENT B (Opponent Rebuttal) ===\n"
            f"Target Claim: {opponent_rebuttal.target_claim}\n"
            f"Counter Arguments: {', '.join(opponent_rebuttal.counter_arguments)}\n"
            f"Flaws Identified: {', '.join(opponent_rebuttal.flaws_identified)}"
        )
        if context and "background" in context:
            user_prompt += f"\nBackground Context: {context['background']}"

        return await self.llm_provider.generate_structured(
            prompt=user_prompt,
            response_model=CriticOutput,
            system_prompt=system_prompt,
        )
