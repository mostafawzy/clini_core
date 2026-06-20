from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from loguru import logger

from .schemas import RAGRequest, RAGResponse
from .service import RAGService, get_rag_service

router = APIRouter(tags=["RAG"])


@router.post("/query", response_model=RAGResponse, summary="Query medical documents")
async def query_documents(
    request: RAGRequest,
    service: RAGService = Depends(get_rag_service),
):

    try:
        return await service.query(
            question=request.query,
            top_k=request.top_k
        )
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post("/upload", summary="Upload a medical PDF")
async def upload_document()


@router.get("/documents", summary="List indexed documents")
async def list_documents()