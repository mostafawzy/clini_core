from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from loguru import logger

from .schemas import RAGRequest, RAGResponse
from .service import RAGService, get_rag_service

router = APIRouter(tags=["RAG"])


@router.post("/query", response_model=RAGResponse, summary="Query medical documents")
async def query_documents(
    
        )


@router.post("/upload", summary="Upload a medical PDF")
async def upload_document()


@router.get("/documents", summary="List indexed documents")
async def list_documents()