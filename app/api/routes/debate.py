from fastapi import APIRouter

from app.api.schemas.debate import DebateRequest, DebateResponse

router = APIRouter()


@router.post("/run", response_model=DebateResponse)
async def run_debate(request: DebateRequest) -> DebateResponse:
    return DebateResponse(
        topic=request.topic,
        winner="Proponent",
        summary=f"Debate completed for topic: {request.topic}",
        arguments=[],
    )
