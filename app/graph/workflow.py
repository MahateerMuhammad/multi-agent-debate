"""LangGraph workflow definition for multi-agent debate."""

from langgraph.graph import StateGraph, END
from app.graph.state import DebateState
from app.graph.nodes import (
    research_node,
    proponent_node,
    opponent_node,
    critic_node,
    judge_node,
)

def create_debate_workflow():
    workflow = StateGraph(DebateState)
    
    workflow.add_node("research", research_node)
    workflow.add_node("proponent", proponent_node)
    workflow.add_node("opponent", opponent_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("judge", judge_node)
    
    workflow.set_entry_point("research")
    workflow.add_edge("research", "proponent")
    workflow.add_edge("proponent", "opponent")
    workflow.add_edge("opponent", "critic")
    workflow.add_edge("critic", "judge")
    workflow.add_edge("judge", END)
    
    return workflow.compile()
