from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from loguru import logger

from .schemas import VisionResponse
from .service import VisionService, get_vision_service

router = APIRouter()

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


@router.post(
    "/classify",
    response_model=VisionResponse,
    summary="Classify a skin lesion image",
)
async def classify_lesion( )