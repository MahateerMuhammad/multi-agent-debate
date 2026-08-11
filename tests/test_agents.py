from app.agents.proponent import ProponentAgent
from app.agents.researcher import ResearcherAgent


def test_researcher_initialization() -> None:
    agent = ResearcherAgent()
    assert agent.name == "Researcher"


def test_proponent_argument() -> None:
    agent = ProponentAgent()
    arg = agent.construct_argument("AI Ethics", [])
    assert "AI Ethics" in arg
