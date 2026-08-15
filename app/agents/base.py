"""Base Agent class defining standard lifecycle, prompt validation, and LLM execution."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.llm.base import BaseLLMProvider
from app.llm.factory import get_llm_provider


class BaseAgent(ABC):
    """Abstract base class for all debate agents."""

    def __init__(
        self,
        name: str,
        role: str,
        description: str,
        llm_provider: BaseLLMProvider | None = None,
    ):
        self.name = name
        self.role = role
        self.description = description
        self._llm_provider = llm_provider

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the concrete agent type identifier."""
        pass

    @property
    def llm_provider(self) -> BaseLLMProvider:
        """Lazy-resolved LLM provider instance."""
        if self._llm_provider is None:
            self._llm_provider = get_llm_provider()
        return self._llm_provider

    def validate_input(self, topic: str) -> str:
        """Validate input topic for non-emptiness, length limits, and guardrails."""
        if not topic or not topic.strip():
            raise ValueError("Debate topic cannot be empty or whitespace.")

        clean_topic = topic.strip()
        
        # Strip literal tags to prevent injection bypass
        clean_topic = clean_topic.replace("<untrusted_input>", "").replace("</untrusted_input>", "")
        
        if len(clean_topic) < 3:
            raise ValueError("Debate topic must be at least 3 characters long.")

        # Delegate prompt guardrail validation to the LLM provider
        self.llm_provider.validate_prompt(clean_topic)
        return clean_topic

    def _format_system_prompt(self, base_instructions: str) -> str:
        """Construct standard system prompt wrapping agent persona and rules."""
        return (
            f"You are {self.name}, an expert AI agent acting as {self.role}.\n"
            f"Role Description: {self.description}\n\n"
            f"Instructions:\n{base_instructions}\n"
            f"Always output strict JSON conforming to the requested schema. "
            f"Be logical, concise, and rigorous."
        )
