"""LangGraph node functions executing Proponent, Opponent, Evidence, Critic, and Judge agents."""

from __future__ import annotations

from typing import Any

from app.agents.critic import CriticAgent
from app.agents.evidence import EvidenceAgent
from app.agents.judge import JudgeAgent
from app.agents.opponent import OpponentAgent
from app.agents.proponent import ProponentAgent
from app.graph.state import DebateState
from app.llm.base import BaseLLMProvider
from app.llm.schemas import ArgumentOutput, CriticOutput, RebuttalOutput
from app.retrieval.retriever import EvidenceRetriever
from app.retrieval.vectorstore import QdrantVectorStore


async def proponent_node(
    state: DebateState,
    llm_provider: BaseLLMProvider | None = None,
) -> dict[str, Any]:
    """Execute Proponent agent node and append argument to state history."""
    topic = state.get("topic", "")
    curr_round = state.get("current_round", 0) + 1
    round_topic = f"{topic} [Debate Round {curr_round}]"
    errors = list(state.get("errors", []))

    try:
        agent = ProponentAgent(llm_provider=llm_provider)
        res = await agent.construct_argument(round_topic)
        arg_dict = res.data.model_dump()

        prop_hist = list(state.get("proponent_history", []))
        prop_hist.append(arg_dict)

        return {
            "proponent_history": prop_hist,
            "total_latency": state.get("total_latency", 0.0) + res.latency_seconds,
            "total_tokens": state.get("total_tokens", 0) + res.usage.total_tokens,
        }
    except Exception as e:
        errors.append(f"proponent_node failure: {e}")
        return {"errors": errors}


async def opponent_node(
    state: DebateState,
    llm_provider: BaseLLMProvider | None = None,
) -> dict[str, Any]:
    """Execute Opponent agent node responding to latest proponent argument."""
    topic = state.get("topic", "")
    curr_round = state.get("current_round", 0) + 1
    round_topic = f"{topic} [Debate Round {curr_round}]"
    errors = list(state.get("errors", []))
    prop_hist = state.get("proponent_history", [])

    if not prop_hist:
        errors.append("opponent_node failure: missing proponent_history")
        return {"errors": errors}

    try:
        latest_prop = ArgumentOutput.model_validate(prop_hist[-1])
        agent = OpponentAgent(llm_provider=llm_provider)
        res = await agent.construct_rebuttal(round_topic, proponent_argument=latest_prop)
        rebuttal_dict = res.data.model_dump()

        opp_hist = list(state.get("opponent_history", []))
        opp_hist.append(rebuttal_dict)

        return {
            "opponent_history": opp_hist,
            "total_latency": state.get("total_latency", 0.0) + res.latency_seconds,
            "total_tokens": state.get("total_tokens", 0) + res.usage.total_tokens,
        }
    except Exception as e:
        errors.append(f"opponent_node failure: {e}")
        return {"errors": errors}


async def evidence_node(
    state: DebateState,
    llm_provider: BaseLLMProvider | None = None,
    vector_store: Any | None = None,
) -> dict[str, Any]:
    """Execute Evidence Verification node auditing claims against Qdrant vector store."""
    topic = state.get("topic", "")
    curr_round = state.get("current_round", 0) + 1
    round_topic = f"{topic} [Debate Round {curr_round}]"
    errors = list(state.get("errors", []))
    prop_hist = state.get("proponent_history", [])


    if not prop_hist:
        errors.append("evidence_node failure: missing proponent_history")
        return {"errors": errors}

    try:
        latest_prop = ArgumentOutput.model_validate(prop_hist[-1])
        store = vector_store or QdrantVectorStore()
        retriever = EvidenceRetriever(vector_store=store)

        search_results = retriever.retrieve_evidence(query=latest_prop.claim, top_k=3)
        context_text = retriever.sanitize_and_wrap_context(search_results)

        agent = EvidenceAgent(llm_provider=llm_provider)
        res = await agent.verify_evidence(
            topic=round_topic,
            claim=latest_prop.claim,
            retrieved_context=context_text,
        )

        ev_dict = res.data.model_dump()

        ev_hist = list(state.get("evidence_history", []))
        ev_hist.append(ev_dict)

        return {
            "evidence_history": ev_hist,
            "total_latency": state.get("total_latency", 0.0) + res.latency_seconds,
            "total_tokens": state.get("total_tokens", 0) + res.usage.total_tokens,
        }
    except Exception as e:
        errors.append(f"evidence_node failure: {e}")
        return {"errors": errors}


async def critic_node(
    state: DebateState,
    llm_provider: BaseLLMProvider | None = None,
) -> dict[str, Any]:
    """Execute position-neutral Critic agent node evaluating proponent & opponent cases."""
    topic = state.get("topic", "")
    curr_round = state.get("current_round", 0) + 1
    round_topic = f"{topic} [Debate Round {curr_round}]"
    errors = list(state.get("errors", []))
    prop_hist = state.get("proponent_history", [])
    opp_hist = state.get("opponent_history", [])

    if not prop_hist or not opp_hist:
        errors.append("critic_node failure: missing proponent or opponent history")
        return {"errors": errors}

    try:
        latest_prop = ArgumentOutput.model_validate(prop_hist[-1])
        latest_opp = RebuttalOutput.model_validate(opp_hist[-1])

        agent = CriticAgent(llm_provider=llm_provider)
        res = await agent.evaluate_debate(
            round_topic, proponent_argument=latest_prop, opponent_rebuttal=latest_opp
        )
        critic_dict = res.data.model_dump()

        crit_hist = list(state.get("critic_history", []))
        crit_hist.append(critic_dict)

        return {
            "critic_history": crit_hist,
            "total_latency": state.get("total_latency", 0.0) + res.latency_seconds,
            "total_tokens": state.get("total_tokens", 0) + res.usage.total_tokens,
        }
    except Exception as e:
        errors.append(f"critic_node failure: {e}")
        return {"errors": errors}


async def judge_node(
    state: DebateState,
    llm_provider: BaseLLMProvider | None = None,
) -> dict[str, Any]:
    """Execute blind Judge agent node scoring debate against 5-dimension rubric."""
    topic = state.get("topic", "")
    curr_round = state.get("current_round", 0) + 1
    round_topic = f"{topic} [Debate Round {curr_round}]"
    errors = list(state.get("errors", []))
    prop_hist = state.get("proponent_history", [])
    opp_hist = state.get("opponent_history", [])
    crit_hist = state.get("critic_history", [])

    if not prop_hist or not opp_hist or not crit_hist:
        errors.append("judge_node failure: missing precursor agent history")
        return {"errors": errors}

    try:
        latest_prop = ArgumentOutput.model_validate(prop_hist[-1])
        latest_opp = RebuttalOutput.model_validate(opp_hist[-1])
        latest_crit = CriticOutput.model_validate(crit_hist[-1])

        position_a_text = (
            f"Claim: {latest_prop.claim}\n"
            f"Reasoning: {', '.join(latest_prop.reasoning)}\n"
            f"Evidence: {', '.join(latest_prop.supporting_evidence)}"
        )
        position_b_text = (
            f"Target Claim: {latest_opp.target_claim}\n"
            f"Counter Arguments: {', '.join(latest_opp.counter_arguments)}\n"
            f"Flaws Identified: {', '.join(latest_opp.flaws_identified)}"
        )

        agent = JudgeAgent(llm_provider=llm_provider)
        res = await agent.judge_debate(
            topic=round_topic,
            position_a_arg=position_a_text,
            position_b_arg=position_b_text,
            critic_evaluation=latest_crit,
        )
        judge_dict = res.data.model_dump()

        jdg_hist = list(state.get("judge_history", []))
        jdg_hist.append(judge_dict)

        return {
            "judge_history": jdg_hist,
            "current_round": curr_round,
            "total_latency": state.get("total_latency", 0.0) + res.latency_seconds,
            "total_tokens": state.get("total_tokens", 0) + res.usage.total_tokens,
        }
    except Exception as e:
        errors.append(f"judge_node failure: {e}")
        return {"errors": errors}
