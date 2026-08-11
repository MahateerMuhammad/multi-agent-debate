"""Judge Agent: Impartial judicial magistrate scoring debates against a 5-dimension rubric."""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.llm.base import BaseLLMProvider
from app.llm.schemas import CriticOutput, JudgeOutput, LLMResponse


class JudgeAgent(BaseAgent):
    """Judge Agent performs impartial blind scoring using a structured 5-dimension rubric."""

    def __init__(
        self,
        name: str = "Judge",
        llm_provider: BaseLLMProvider | None = None,
    ):
        super().__init__(
            name=name,
            role="Impartial Judicial Magistrate",
            description=(
                "Evaluates Position A and Position B arguments against a 5-dimension rubric "
                "(correctness, evidence_quality, reasoning, relevance, completeness)."
            ),
            llm_provider=llm_provider,
        )

    @property
    def agent_type(self) -> str:
        return "judge"

    async def judge_debate(
        self,
        topic: str,
        position_a_arg: str,
        position_b_arg: str,
        critic_evaluation: CriticOutput,
        context: dict[str, Any] | None = None,
    ) -> LLMResponse[JudgeOutput]:
        """Scoring anonymized Position A and Position B to declare an impartial winner."""
        clean_topic = self.validate_input(topic)

        system_prompt = self._format_system_prompt(
            "You are an objective judicial magistrate presiding over an anonymized debate.\n"
            "You must score Position A and Position B against the following 5 dimensions:\n"
            "1. correctness: Factual accuracy and absence of misinformation.\n"
            "2. evidence_quality: Strength, relevance, and credibility of evidence.\n"
            "3. reasoning: Logical validity, coherence, and absence of fallacies.\n"
            "4. relevance: Direct responsiveness to the core proposition topic.\n"
            "5. completeness: Thoroughness of arguments and address of counterpoints.\n\n"
            "For EACH dimension, provide numerical scores (0.0 to 1.0) and justification.\n"
            "Do NOT declare a winner without explaining scoring and key deciding factors.\n"
            "Winner must be explicitly declared as 'Position A', 'Position B', or 'Tie'."
        )

        user_prompt = (
            f"Debate Proposition Topic: {clean_topic}\n\n"
            f"=== ANONYMIZED POSITION A ===\n{position_a_arg}\n\n"
            f"=== ANONYMIZED POSITION B ===\n{position_b_arg}\n\n"
            f"=== INDEPENDENT CRITIC AUDIT ===\n"
            f"Analysis of A: {critic_evaluation.argument_a_analysis}\n"
            f"Analysis of B: {critic_evaluation.argument_b_analysis}\n"
            f"Unsupported Claims Identified: {', '.join(critic_evaluation.unsupported_claims)}\n"
            f"Logical Fallacies Identified: {', '.join(critic_evaluation.logical_fallacies)}\n"
            f"Missing Assumptions: {', '.join(critic_evaluation.missing_assumptions)}\n"
            f"Contradictions Found: {', '.join(critic_evaluation.contradictions_found)}"
        )
        if context and "background" in context:
            user_prompt += f"\nBackground Context: {context['background']}"

        return await self.llm_provider.generate_structured(
            prompt=user_prompt,
            response_model=JudgeOutput,
            system_prompt=system_prompt,
        )
