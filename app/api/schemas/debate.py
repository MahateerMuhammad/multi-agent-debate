from pydantic import BaseModel, Field
from typing import List, Dict, Any

class DebateRequest(BaseModel):
    topic: str = Field(..., description="The debate topic or proposition")
    rounds: int = Field(default=3, ge=1, le=10, description="Number of debate rounds")

class DebateResponse(BaseModel):
    topic: str
    winner: str
    summary: str
    arguments: List[Dict[str, Any]]
