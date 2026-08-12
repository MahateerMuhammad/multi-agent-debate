"""Assembly and compilation of the LangGraph multi-agent debate workflow graph."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph.edges import should_continue
from app.graph.nodes import critic_node, judge_node, opponent_node, proponent_node
from app.graph.state import DebateState
from app.llm.base import BaseLLMProvider


def build_debate_graph(llm_provider: BaseLLMProvider | None = None) -> Any:
    """Build and compile the multi-agent debate LangGraph workflow."""
    builder = StateGraph(DebateState)

    # Bind provider parameter to node functions
    prop_fn = partial(proponent_node, llm_provider=llm_provider)
    opp_fn = partial(opponent_node, llm_provider=llm_provider)
    crit_fn = partial(critic_node, llm_provider=llm_provider)
    jdg_fn = partial(judge_node, llm_provider=llm_provider)

    builder.add_node("proponent", prop_fn)
    builder.add_node("opponent", opp_fn)
    builder.add_node("critic", crit_fn)
    builder.add_node("judge", jdg_fn)

    builder.add_edge(START, "proponent")
    builder.add_edge("proponent", "opponent")
    builder.add_edge("opponent", "critic")
    builder.add_edge("critic", "judge")

    builder.add_conditional_edges(
        "judge",
        should_continue,
        {
            "proponent": "proponent",
            "end": END,
        },
    )

    return builder.compile()
