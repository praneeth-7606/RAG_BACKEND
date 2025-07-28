from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class UploadStatus(str, Enum):
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"

class DocumentChunk(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class UploadResponse(BaseModel):
    status: UploadStatus
    message: str
    document_id: Optional[str] = None
    chunks_count: Optional[int] = None

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)

class Source(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    relevance_score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    query: str

class ErrorResponse(BaseModel):
    detail: str
    error_type: Optional[str] = None