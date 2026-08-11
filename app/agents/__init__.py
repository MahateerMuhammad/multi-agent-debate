"""Debate Agents package."""

from app.agents.base import BaseAgent
from app.agents.critic import CriticAgent
from app.agents.evidence import EvidenceAgent
from app.agents.judge import JudgeAgent
from app.agents.opponent import OpponentAgent
from app.agents.proponent import ProponentAgent
from app.agents.researcher import ResearcherAgent
from app.agents.runner import SimpleDebateRunner
from app.agents.schemas import SimpleDebateResult

__all__ = [
    "BaseAgent",
    "ProponentAgent",
    "OpponentAgent",
    "ResearcherAgent",
    "CriticAgent",
    "EvidenceAgent",
    "JudgeAgent",
    "SimpleDebateRunner",
    "SimpleDebateResult",
]
