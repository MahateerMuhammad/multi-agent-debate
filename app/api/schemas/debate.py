from typing import Any

from pydantic import BaseModel, Field


class DebateRequest(BaseModel):
    topic: str = Field(..., max_length=500, description="The debate topic or proposition")
    rounds: int = Field(default=3, ge=1, le=10, description="Number of debate rounds")


class DebateResponse(BaseModel):
    topic: str
    winner: str
    summary: str
    arguments: list[dict[str, Any]]
