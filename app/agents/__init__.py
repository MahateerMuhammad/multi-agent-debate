"""Debate Agents package."""

from app.agents.base import BaseAgent
from app.agents.critic import CriticAgent
from app.agents.evidence import EvidenceAgent
from app.agents.judge import JudgeAgent
from app.agents.opponent import OpponentAgent
from app.agents.proponent import ProponentAgent
from app.agents.researcher import ResearcherAgent
from app.agents.runner import FullDebateRunner, SimpleDebateRunner
from app.agents.schemas import FullDebateResult, SimpleDebateResult

__all__ = [
    "BaseAgent",
    "ProponentAgent",
    "OpponentAgent",
    "CriticAgent",
    "JudgeAgent",
    "ResearcherAgent",
    "EvidenceAgent",
    "SimpleDebateRunner",
    "FullDebateRunner",
    "SimpleDebateResult",
    "FullDebateResult",
]
