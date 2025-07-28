# # app/api/routes.py
# import os
# import uuid
# import aiofiles
# from pathlib import Path
# from fastapi import APIRouter, UploadFile, File, HTTPException, Form
# from fastapi.responses import JSONResponse
# from loguru import logger

# from app.core.config import settings
# from app.models.schemas import (
#     UploadResponse, UploadStatus, ChatRequest, ChatResponse, ErrorResponse
# )
# from app.services.document_processor import DocumentProcessor
# from app.services.vector_store import VectorStore
# from app.services.llm_service import LLMService
# from app.core.embeddings import EmbeddingService

# router = APIRouter()

# # Initialize services (create them when needed to avoid import issues)
# _embedding_service = None
# _vector_store = None
# _document_processor = None
# _llm_service = None

# def get_embedding_service():
#     global _embedding_service
#     if _embedding_service is None:
#         _embedding_service = EmbeddingService()
#     return _embedding_service

# def get_vector_store():
#     global _vector_store
#     if _vector_store is None:
#         _vector_store = VectorStore()
#     return _vector_store

# def get_document_processor():
#     global _document_processor
#     if _document_processor is None:
#         _document_processor = DocumentProcessor()
#     return _document_processor

# def get_llm_service():
#     global _llm_service
#     if _llm_service is None:
#         _llm_service = LLMService()
#     return _llm_service

# @router.post("/upload", response_model=UploadResponse)
# async def upload_document(file: UploadFile = File(...)):
#     """Upload and process insurance document"""
#     file_path = None  # Initialize for cleanup
    
#     try:
#         # Validate file type
#         file_extension = Path(file.filename).suffix.lower()
#         if file_extension not in settings.SUPPORTED_EXTENSIONS:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Unsupported file type. Supported: {settings.SUPPORTED_EXTENSIONS}"
#             )
        
#         # Validate file size
#         content = await file.read()
#         file_size = len(content)
        
#         if file_size > settings.MAX_FILE_SIZE * 1024 * 1024:  # Convert MB to bytes
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE}MB"
#             )
        
#         # Save uploaded file
#         file_id = str(uuid.uuid4())
#         safe_filename = f"{file_id}_{file.filename}"
#         file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
        
#         async with aiofiles.open(file_path, 'wb') as f:
#             await f.write(content)
        
#         # Get services
#         document_processor = get_document_processor()
#         vector_store = get_vector_store()
#         embedding_service = get_embedding_service()
        
#         try:
#             # Process document (this will include insurance validation)
#             chunks, doc_id = await document_processor.process_file(file_path, file.filename)
            
#             # Add to vector store
#             await vector_store.add_chunks(chunks, embedding_service)
            
#             # Clean up uploaded file (optional)
#             # os.remove(file_path)
            
#             return UploadResponse(
#                 status=UploadStatus.SUCCESS,
#                 message="Insurance document processed successfully and validated as insurance-related content.",
#                 document_id=doc_id,
#                 chunks_count=len(chunks)
#             )
            
#         except ValueError as ve:
#             # This is our insurance validation error
#             logger.warning(f"Document validation failed: {str(ve)}")
            
#             # Clean up uploaded file
#             if file_path and os.path.exists(file_path):
#                 os.remove(file_path)
            
#             return UploadResponse(
#                 status=UploadStatus.FAILED,
#                 message=str(ve)
#             )
    
#     except HTTPException:
#         # Clean up uploaded file if exists
#         if file_path and os.path.exists(file_path):
#             os.remove(file_path)
#         raise
        
#     except Exception as e:
#         logger.error(f"Upload error: {e}")
        
#         # Clean up uploaded file if exists
#         if file_path and os.path.exists(file_path):
#             os.remove(file_path)
            
#         return UploadResponse(
#             status=UploadStatus.FAILED,
#             message=f"Failed to process document: {str(e)}"
#         )

# @router.post("/chat", response_model=ChatResponse)
# async def chat_with_documents(request: ChatRequest):
#     """Ask questions about uploaded documents"""
#     try:
#         # Get services
#         vector_store = get_vector_store()
#         embedding_service = get_embedding_service()
#         llm_service = get_llm_service()
        
#         # Search for relevant sources
#         sources = await vector_store.search_similar(request.query, embedding_service, k=5)
        
#         # Generate answer using LLM (includes query validation)
#         answer = await llm_service.generate_answer(request.query, sources)
        
#         return ChatResponse(
#             answer=answer,
#             sources=sources,
#             query=request.query
#         )
    
#     except Exception as e:
#         logger.error(f"Chat error: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail="Failed to process your question"
#         )

# @router.get("/status")
# async def get_status():
#     """Get API status and statistics"""
#     try:
#         vector_store = get_vector_store()
#         total_chunks = len(vector_store.chunks) if vector_store.chunks else 0
        
#         return {
#             "status": "healthy",
#             "total_chunks": total_chunks,
#             "supported_formats": settings.SUPPORTED_EXTENSIONS,
#             "max_file_size_mb": settings.MAX_FILE_SIZE,
#             "service_type": "Insurance Policy Q&A System",
#             "validation": "AI-powered insurance content validation enabled"
#         }
    
#     except Exception as e:
#         logger.error(f"Status error: {e}")
#         return {"status": "error", "message": str(e)}



# app/api/routes.py - Updated with documents list endpoint
import os
import uuid
import aiofiles
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.config import settings
from app.models.schemas import (
    UploadResponse, UploadStatus, ChatRequest, ChatResponse, ErrorResponse
)
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.core.embeddings import EmbeddingService

router = APIRouter()

# Initialize services (create them when needed to avoid import issues)
_embedding_service = None
_vector_store = None
_document_processor = None
_llm_service = None

def get_embedding_service():
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store

def get_document_processor():
    global _document_processor
    if _document_processor is None:
        _document_processor = DocumentProcessor()
    return _document_processor

def get_llm_service():
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process insurance document"""
    file_path = None  # Initialize for cleanup
    
    try:
        # Validate file type
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: {settings.SUPPORTED_EXTENSIONS}"
            )
        
        # Validate file size
        content = await file.read()
        file_size = len(content)
        
        if file_size > settings.MAX_FILE_SIZE * 1024 * 1024:  # Convert MB to bytes
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE}MB"
            )
        
        # Save uploaded file
        file_id = str(uuid.uuid4())
        safe_filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(settings.UPLOAD_DIR, safe_filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        # Get services
        document_processor = get_document_processor()
        vector_store = get_vector_store()
        embedding_service = get_embedding_service()
        
        try:
            # Process document (this will include insurance validation)
            chunks, doc_id = await document_processor.process_file(file_path, file.filename)
            
            # Prepare document info for vector store
            document_info = {
                'filename': file.filename,
                'file_size': file_size,
                'document_id': doc_id
            }
            
            # Add to vector store with document info
            await vector_store.add_chunks(chunks, embedding_service, document_info)
            
            # Clean up uploaded file (optional)
            # os.remove(file_path)
            
            return UploadResponse(
                status=UploadStatus.SUCCESS,
                message="Insurance document processed successfully and validated as insurance-related content.",
                document_id=doc_id,
                chunks_count=len(chunks)
            )
            
        except ValueError as ve:
            # This is our insurance validation error
            logger.warning(f"Document validation failed: {str(ve)}")
            
            # Clean up uploaded file
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
            
            return UploadResponse(
                status=UploadStatus.FAILED,
                message=str(ve)
            )
    
    except HTTPException:
        # Clean up uploaded file if exists
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        
        # Clean up uploaded file if exists
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            
        return UploadResponse(
            status=UploadStatus.FAILED,
            message=f"Failed to process document: {str(e)}"
        )

@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(request: ChatRequest):
    """Ask questions about uploaded documents"""
    try:
        # Get services
        vector_store = get_vector_store()
        embedding_service = get_embedding_service()
        llm_service = get_llm_service()
        
        # Search for relevant sources
        sources = await vector_store.search_similar(request.query, embedding_service, k=5)
        
        # Generate answer using LLM (includes query validation)
        answer = await llm_service.generate_answer(request.query, sources)
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            query=request.query
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process your question"
        )

# NEW: Get all uploaded documents
@router.get("/documents")
async def get_all_documents():
    """Get list of all uploaded documents"""
    try:
        vector_store = get_vector_store()
        documents = vector_store.get_all_documents()
        
        return {
            "documents": documents,
            "total_count": len(documents)
        }
    
    except Exception as e:
        logger.error(f"Failed to get documents: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve documents list"
        )

@router.get("/status")
async def get_status():
    """Get API status and statistics"""
    try:
        vector_store = get_vector_store()
        stats = vector_store.get_document_stats()
        
        return {
            "status": "healthy",
            "total_documents": stats['total_documents'],
            "total_chunks": stats['total_chunks'],
            "average_chunks_per_document": stats['average_chunks_per_document'],
            "supported_formats": settings.SUPPORTED_EXTENSIONS,
            "max_file_size_mb": settings.MAX_FILE_SIZE,
            "service_type": "Insurance Policy Q&A System",
            "validation": "AI-powered insurance content validation enabled"
        }
    
    except Exception as e:
        logger.error(f"Status error: {e}")
        return {"status": "error", "message": str(e)}