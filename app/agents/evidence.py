"""Evidence Verification Agent auditing claim factuality against retrieved document context."""

from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.llm.base import BaseLLMProvider
from app.llm.schemas import EvidenceVerificationOutput, LLMResponse


class EvidenceAgent(BaseAgent):
    """Evidence Agent audits debate claims against retrieved vector store context."""

    def __init__(
        self,
        name: str = "EvidenceVerifier",
        llm_provider: BaseLLMProvider | None = None,
    ):
        super().__init__(
            name=name,
            role="Evidence Verification Auditor",
            description=(
                "Audits debate claims against retrieved vector store context and classifies status "
                "into 4 categories (supported, contradicted, etc)."
            ),
            llm_provider=llm_provider,
        )

    @property
    def agent_type(self) -> str:
        return "evidence"

    async def verify_evidence(
        self,
        topic: str,
        claim: str,
        retrieved_context: str,
        context: dict[str, Any] | None = None,
    ) -> LLMResponse[EvidenceVerificationOutput]:
        """Audit claim against retrieved evidence context and return 4-way fact grounding status."""
        clean_topic = self.validate_input(topic)

        system_prompt = self._format_system_prompt(
            "You are a strict, objective Evidence Verification Auditor.\n"
            "Compare the asserted claim against retrieved source document context.\n"
            "Classify the status into EXACTLY ONE of the 4 categories:\n"
            "1. 'supported': Directly confirmed by facts in retrieved context.\n"
            "2. 'partially_supported': Mostly true but has minor caveats.\n"
            "3. 'contradicted': Directly disproven by retrieved context.\n"
            "4. 'insufficient_evidence': Context does not contain enough facts to verify.\n\n"
            "Set is_verified = true ONLY if status is 'supported' or 'partially_supported'.\n"
            "Extract source titles or URLs into sources_cited."
        )

        user_prompt = (
            f"Debate Proposition Topic: {clean_topic}\n\n"
            f"ASSERTED CLAIM TO VERIFY:\n{claim}\n\n"
            f"RETRIEVED VECTOR CONTEXT:\n{retrieved_context}"
        )
        if context and "background" in context:
            user_prompt += f"\nBackground Context: {context['background']}"

        return await self.llm_provider.generate_structured(
            prompt=user_prompt,
            response_model=EvidenceVerificationOutput,
            system_prompt=system_prompt,
        )
