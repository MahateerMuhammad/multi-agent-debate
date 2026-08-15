"""Debate Runners coordinating sequential Proponent, Opponent, Critic, and Judge execution."""

from __future__ import annotations

from typing import Any

from app.agents.critic import CriticAgent
from app.agents.judge import JudgeAgent
from app.agents.opponent import OpponentAgent
from app.agents.proponent import ProponentAgent
from app.agents.schemas import FullDebateResult, SimpleDebateResult
from app.llm.base import BaseLLMProvider


class SimpleDebateRunner:
    """Coordinator executing a 2-agent sequential debate session (Proponent vs Opponent)."""

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
        proponent_res = await self.proponent.construct_argument(topic, context=context)
        opponent_res = await self.opponent.construct_rebuttal(
            topic, proponent_argument=proponent_res.data, context=context
        )

        total_latency = proponent_res.latency_seconds + opponent_res.latency_seconds
        total_tokens = (proponent_res.usage.total_tokens or 0) + (
            opponent_res.usage.total_tokens or 0
        )

        return SimpleDebateResult(
            topic=topic.strip(),
            proponent_output=proponent_res.data,
            opponent_output=opponent_res.data,
            total_latency_seconds=total_latency,
            total_tokens=total_tokens,
        )


class FullDebateRunner:
    """Coordinator executing a complete 4-agent debate pipeline with blind rubric judging."""

    def __init__(
        self,
        proponent: ProponentAgent | None = None,
        opponent: OpponentAgent | None = None,
        critic: CriticAgent | None = None,
        judge: JudgeAgent | None = None,
        llm_provider: BaseLLMProvider | None = None,
    ):
        self.proponent = proponent or ProponentAgent(llm_provider=llm_provider)
        self.opponent = opponent or OpponentAgent(llm_provider=llm_provider)
        self.critic = critic or CriticAgent(llm_provider=llm_provider)
        self.judge = judge or JudgeAgent(llm_provider=llm_provider)

    async def run_full_debate(
        self,
        topic: str,
        context: dict[str, Any] | None = None,
    ) -> FullDebateResult:
        """Run 4-stage pipeline: Proponent -> Opponent -> Critic -> Blind Judge."""
        # 1. Proponent Argument
        prop_res = await self.proponent.construct_argument(topic, context=context)

        # 2. Opponent Rebuttal
        opp_res = await self.opponent.construct_rebuttal(
            topic, proponent_argument=prop_res.data, context=context
        )

        # 3. Position-Neutral Critic Analysis
        critic_res = await self.critic.evaluate_debate(
            topic, proponent_argument=prop_res.data, opponent_rebuttal=opp_res.data, context=context
        )

        # 4. Blind Anonymization Mapping
        blind_mapping = {"Position A": "Proponent", "Position B": "Opponent"}

        position_a_text = (
            f"Claim: {prop_res.data.claim}\n"
            f"Reasoning: {', '.join(prop_res.data.reasoning)}\n"
            f"Evidence: {', '.join(prop_res.data.supporting_evidence)}"
        )
        position_b_text = (
            f"Target Claim: {opp_res.data.target_claim}\n"
            f"Counter Arguments: {', '.join(opp_res.data.counter_arguments)}\n"
            f"Flaws Identified: {', '.join(opp_res.data.flaws_identified)}"
        )

        # 5. Judge Evaluation with Rubric Scoring
        judge_res = await self.judge.judge_debate(
            topic=topic,
            position_a_arg=position_a_text,
            position_b_arg=position_b_text,
            critic_evaluation=critic_res.data,
            context=context,
        )

        # 6. Unblind Verdict Mapping
        raw_winner = judge_res.data.winner.strip()
        unblinded_winner = blind_mapping.get(raw_winner, raw_winner)
        if unblinded_winner not in ("Proponent", "Opponent", "Tie"):
            unblinded_winner = (
                "Proponent" if "A" in raw_winner else ("Opponent" if "B" in raw_winner else "Tie")
            )

        total_latency = (
            prop_res.latency_seconds
            + opp_res.latency_seconds
            + critic_res.latency_seconds
            + judge_res.latency_seconds
        )
        total_tokens = (
            (prop_res.usage.total_tokens or 0)
            + (opp_res.usage.total_tokens or 0)
            + (critic_res.usage.total_tokens or 0)
            + (judge_res.usage.total_tokens or 0)
        )

        return FullDebateResult(
            topic=topic.strip(),
            proponent_output=prop_res.data,
            opponent_output=opp_res.data,
            critic_output=critic_res.data,
            judge_output=judge_res.data,
            blind_mapping=blind_mapping,
            unblinded_winner=unblinded_winner,
            total_latency_seconds=total_latency,
            total_tokens=total_tokens,
        )

    async def run_full_debate_stream(
        self,
        topic: str,
        context: dict[str, Any] | None = None,
    ) -> Any:
        """Stream 4-stage pipeline events as Server-Sent Events (JSON strings)."""
        import json
        from collections.abc import AsyncGenerator

        async def _stream() -> AsyncGenerator[str, None]:
            # 1. Proponent
            yield json.dumps({"event": "status", "data": "Proponent is researching and drafting argument..."}) + "\n"
            prop_res = await self.proponent.construct_argument(topic, context=context)
            yield json.dumps({"event": "proponent", "data": prop_res.data.model_dump()}) + "\n"

            # 2. Opponent
            yield json.dumps({"event": "status", "data": "Opponent is analyzing claim and drafting rebuttal..."}) + "\n"
            opp_res = await self.opponent.construct_rebuttal(
                topic, proponent_argument=prop_res.data, context=context
            )
            yield json.dumps({"event": "opponent", "data": opp_res.data.model_dump()}) + "\n"

            # 3. Critic
            yield json.dumps({"event": "status", "data": "Critic is evaluating logical fallacies..."}) + "\n"
            critic_res = await self.critic.evaluate_debate(
                topic, proponent_argument=prop_res.data, opponent_rebuttal=opp_res.data, context=context
            )
            yield json.dumps({"event": "critic", "data": critic_res.data.model_dump()}) + "\n"

            # 4. Judge
            yield json.dumps({"event": "status", "data": "Judge is assigning blind rubric scores..."}) + "\n"
            blind_mapping = {"Position A": "Proponent", "Position B": "Opponent"}
            position_a_text = f"Claim: {prop_res.data.claim}\nReasoning: {', '.join(prop_res.data.reasoning)}\nEvidence: {', '.join(prop_res.data.supporting_evidence)}"
            position_b_text = f"Target Claim: {opp_res.data.target_claim}\nCounter Arguments: {', '.join(opp_res.data.counter_arguments)}\nFlaws Identified: {', '.join(opp_res.data.flaws_identified)}"

            judge_res = await self.judge.judge_debate(
                topic=topic,
                position_a_arg=position_a_text,
                position_b_arg=position_b_text,
                critic_evaluation=critic_res.data,
                context=context,
            )
            
            raw_winner = judge_res.data.winner.strip()
            unblinded_winner = blind_mapping.get(raw_winner, raw_winner)
            if unblinded_winner not in ("Proponent", "Opponent", "Tie"):
                unblinded_winner = "Proponent" if "A" in raw_winner else ("Opponent" if "B" in raw_winner else "Tie")

            final_result = {
                "winner": unblinded_winner,
                "scores": judge_res.data.model_dump(),
                "total_latency": prop_res.latency_seconds + opp_res.latency_seconds + critic_res.latency_seconds + judge_res.latency_seconds,
                "total_tokens": (prop_res.usage.total_tokens or 0) + (opp_res.usage.total_tokens or 0) + (critic_res.usage.total_tokens or 0) + (judge_res.usage.total_tokens or 0)
            }
            yield json.dumps({"event": "judge", "data": final_result}) + "\n"

        return _stream()
