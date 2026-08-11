"""Simple Debate Runner coordinating sequential Proponent vs Opponent execution."""

from __future__ import annotations

from typing import Any

from app.agents.opponent import OpponentAgent
from app.agents.proponent import ProponentAgent
from app.agents.schemas import SimpleDebateResult
from app.llm.base import BaseLLMProvider


class SimpleDebateRunner:
    """Coordinator executing a 2-agent sequential debate session."""

    def __init__(
        self,
        proponent: ProponentAgent | None = None,
        opponent: OpponentAgent | None = None,
        llm_provider: BaseLLMProvider | None = None,
    ):
        self.proponent = proponent or ProponentAgent(llm_provider=llm_provider)
        self.opponent = opponent or OpponentAgent(llm_provider=llm_provider)

    async def run_debate(
        self,
        topic: str,
        context: dict[str, Any] | None = None,
    ) -> SimpleDebateResult:
        """Run a full 2-agent debate sequence for the given topic proposition."""
        # 1. Proponent constructs argument
        proponent_res = await self.proponent.construct_argument(topic, context=context)

        # 2. Opponent constructs counter-argument rebuttal
        opponent_res = await self.opponent.construct_rebuttal(
            topic, proponent_argument=proponent_res.data, context=context
        )

        total_latency = proponent_res.latency_seconds + opponent_res.latency_seconds
        total_tokens = proponent_res.usage.total_tokens + opponent_res.usage.total_tokens

        return SimpleDebateResult(
            topic=topic.strip(),
            proponent_output=proponent_res.data,
            opponent_output=opponent_res.data,
            total_latency_seconds=total_latency,
            total_tokens=total_tokens,
        )
