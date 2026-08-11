from fastapi import APIRouter
from app.api.routes.debate import router as debate_router

router = APIRouter()
router.include_router(debate_router, prefix="/debate", tags=["debate"])
