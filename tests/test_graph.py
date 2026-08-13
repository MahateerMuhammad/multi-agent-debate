from __future__ import annotations

import pytest

from app.graph.edges import should_continue
from app.graph.nodes import critic_node, judge_node, opponent_node, proponent_node
from app.graph.state import DebateState
from app.graph.workflow import build_debate_graph
from app.llm.providers.mock import MockLLMProvider


def test_should_continue_max_rounds() -> None:
    """Test that should_continue returns 'end' when current_round >= max_rounds."""
    state: DebateState = {
        "current_round": 3,
        "max_rounds": 3,
        "errors": [],
    }
    assert should_continue(state) == "end"


def test_should_continue_error_safeguard() -> None:
    """Test that should_continue returns 'end' when state contains errors."""
    state: DebateState = {
        "current_round": 1,
        "max_rounds": 3,
        "errors": ["proponent_node failure: network timeout"],
    }
    assert should_continue(state) == "end"


def test_should_continue_confidence_threshold_met() -> None:
    """Test that should_continue returns 'end' when Judge score meets confidence threshold."""
    state: DebateState = {
        "current_round": 1,
        "max_rounds": 3,
        "confidence_threshold": 0.90,
        "judge_history": [{"total_score_a": 0.95, "total_score_b": 0.70}],
        "errors": [],
    }
    assert should_continue(state) == "end"


def test_should_continue_low_improvement_threshold() -> None:
    """Test that should_continue returns 'end' when score improvement is below threshold."""
    state: DebateState = {
        "current_round": 2,
        "max_rounds": 3,
        "confidence_threshold": 0.95,
        "improvement_threshold": 0.05,
        "judge_history": [
            {"total_score_a": 0.80, "total_score_b": 0.70},
            {"total_score_a": 0.81, "total_score_b": 0.71},  # delta = 0.01 < 0.05
        ],
        "errors": [],
    }
    assert should_continue(state) == "end"


def test_should_continue_zero_improvement_stops() -> None:
    """Test that should_continue returns 'end' when score delta is 0.0."""
    state: DebateState = {
        "current_round": 2,
        "max_rounds": 3,
        "confidence_threshold": 0.95,
        "improvement_threshold": 0.01,
        "judge_history": [
            {"total_score_a": 0.80, "total_score_b": 0.70},
            {"total_score_a": 0.80, "total_score_b": 0.70},  # delta = 0.0
        ],
        "errors": [],
    }
    assert should_continue(state) == "end"


def test_should_continue_loops_to_proponent() -> None:
    """Test that should_continue loops back to 'proponent' when improvement is sufficient."""
    state: DebateState = {
        "current_round": 1,
        "max_rounds": 3,
        "confidence_threshold": 0.95,
        "improvement_threshold": 0.02,
        "judge_history": [{"total_score_a": 0.70, "total_score_b": 0.65}],
        "errors": [],
    }
    assert should_continue(state) == "proponent"


@pytest.mark.asyncio
async def test_opponent_node_missing_precursor_error() -> None:
    """Test that opponent_node handles missing proponent history gracefully."""
    empty_state: DebateState = {"topic": "AI Judges", "proponent_history": [], "errors": []}
    res = await opponent_node(empty_state, llm_provider=MockLLMProvider())
    assert "errors" in res
    assert "missing proponent_history" in res["errors"][0]


@pytest.mark.asyncio
async def test_critic_node_missing_precursor_error() -> None:
    """Test that critic_node handles missing precursor histories gracefully."""
    empty_state: DebateState = {
        "topic": "AI Judges",
        "proponent_history": [],
        "opponent_history": [],
        "errors": [],
    }
    res = await critic_node(empty_state, llm_provider=MockLLMProvider())
    assert "errors" in res
    assert "missing proponent or opponent history" in res["errors"][0]


@pytest.mark.asyncio
async def test_judge_node_missing_precursor_error() -> None:
    """Test that judge_node handles missing precursor histories gracefully."""
    empty_state: DebateState = {
        "topic": "AI Judges",
        "proponent_history": [],
        "opponent_history": [],
        "critic_history": [],
        "errors": [],
    }
    res = await judge_node(empty_state, llm_provider=MockLLMProvider())
    assert "errors" in res
    assert "missing precursor agent history" in res["errors"][0]


@pytest.mark.asyncio
async def test_proponent_node_execution() -> None:
    """Test individual proponent_node execution."""
    mock_llm = MockLLMProvider()
    state: DebateState = {"topic": "Open source AI", "proponent_history": [], "errors": []}
    res = await proponent_node(state, llm_provider=mock_llm)

    assert "proponent_history" in res
    assert len(res["proponent_history"]) == 1
    assert res["total_tokens"] > 0


@pytest.mark.asyncio
async def test_langgraph_end_to_end_execution() -> None:
    """Test full LangGraph workflow execution using MockLLMProvider."""
    mock_llm = MockLLMProvider()
    graph = build_debate_graph(llm_provider=mock_llm)

    initial_state: DebateState = {
        "topic": "Universal basic income should be implemented globally",
        "current_round": 0,
        "max_rounds": 2,
        "confidence_threshold": 0.90,
        "improvement_threshold": 0.02,
        "proponent_history": [],
        "opponent_history": [],
        "critic_history": [],
        "judge_history": [],
        "errors": [],
    }

    final_state = await graph.ainvoke(initial_state)

    assert final_state["current_round"] > 0
    assert len(final_state["proponent_history"]) > 0
    assert len(final_state["opponent_history"]) > 0
    assert len(final_state["critic_history"]) > 0
    assert len(final_state["judge_history"]) > 0
    assert final_state["total_tokens"] > 0


@pytest.mark.asyncio
async def test_graph_state_invariants() -> None:
    """Test monotonic increment of current_round and relation integrity."""
    mock_llm = MockLLMProvider()
    graph = build_debate_graph(llm_provider=mock_llm)

    initial_state: DebateState = {
        "topic": "Graph Invariants Test",
        "current_round": 0,
        "max_rounds": 2,
        "confidence_threshold": 0.99,  # Force multiple rounds
        "improvement_threshold": 0.001,
        "proponent_history": [],
        "opponent_history": [],
        "evidence_history": [],
        "critic_history": [],
        "judge_history": [],
        "errors": [],
    }

    final_state = await graph.ainvoke(initial_state)

    # Invariants:
    # 1. Monotonic increment of current_round
    assert final_state["current_round"] == 2

    # 2. History lists should all have length == current_round
    rounds_executed = final_state["current_round"]
    assert len(final_state["proponent_history"]) == rounds_executed
    assert len(final_state["opponent_history"]) == rounds_executed
    assert len(final_state["evidence_history"]) == rounds_executed
    assert len(final_state["critic_history"]) == rounds_executed
    assert len(final_state["judge_history"]) == rounds_executed

    # 3. No fatal errors masquerading as normal state
    assert not final_state.get("errors")
