from app.agents.researcher import ResearcherAgent
from app.agents.proponent import ProponentAgent
from app.agents.opponent import OpponentAgent

def test_researcher_initialization():
    agent = ResearcherAgent()
    assert agent.name == "Researcher"

def test_proponent_argument():
    agent = ProponentAgent()
    arg = agent.construct_argument("AI Ethics", [])
    assert "AI Ethics" in arg
