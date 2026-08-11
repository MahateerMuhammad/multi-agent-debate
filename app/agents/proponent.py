"""Proponent Agent: Formulates affirmative arguments supporting a topic proposition."""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.llm.base import BaseLLMProvider
from app.llm.schemas import ArgumentOutput, LLMResponse


class ProponentAgent(BaseAgent):
    """Proponent Agent constructs compelling, structured affirmative arguments."""

    def __init__(
        self,
        name: str = "Proponent",
        llm_provider: BaseLLMProvider | None = None,
    ):
        super().__init__(
            name=name,
            role="Affirmative Debate Proponent",
            description="Constructs logical, persuasive arguments supporting the proposition.",
            llm_provider=llm_provider,
        )

    @property
    def agent_type(self) -> str:
        return "proponent"

    async def construct_argument(
        self,
        topic: str,
        context: dict[str, Any] | None = None,
    ) -> LLMResponse[ArgumentOutput]:
        """Construct a structured argument supporting the given proposition."""
        clean_topic = self.validate_input(topic)

        system_prompt = self._format_system_prompt(
            "Your objective is to construct the strongest affirmative argument for the topic.\n"
            "Provide a clear central claim, structured step-by-step reasoning points, "
            "and supporting evidence."
        )

        user_prompt = f"Proposition Topic: {clean_topic}"
        if context and "background" in context:
            user_prompt += f"\nBackground Context: {context['background']}"

        return await self.llm_provider.generate_structured(
            prompt=user_prompt,
            response_model=ArgumentOutput,
            system_prompt=system_prompt,
        )
