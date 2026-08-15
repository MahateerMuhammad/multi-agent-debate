import asyncio
from fastapi import APIRouter, Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.api.schemas.debate import DebateRequest, DebateResponse

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/run", response_model=DebateResponse)
@limiter.limit("5/minute")
async def run_debate(request: Request, body: DebateRequest) -> DebateResponse:
    async def _execute():
        return DebateResponse(
            topic=body.topic,
            winner="Proponent",
            summary=f"Debate completed for topic: {body.topic}",
            arguments=[],
        )
        
    try:
        return await asyncio.wait_for(_execute(), timeout=300.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Debate execution timed out.")
