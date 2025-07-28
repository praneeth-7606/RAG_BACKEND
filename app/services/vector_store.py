# import os
# import pickle
# import numpy as np
# import faiss
# from typing import List, Tuple, Set
# from loguru import logger
# from difflib import SequenceMatcher

# from app.core.config import settings
# from app.models.schemas import DocumentChunk, Source

# class VectorStore:
#     def __init__(self):
#         self.index = None
#         self.chunks = []
#         self.dimension = 384  # all-MiniLM-L6-v2 dimension
#         self.index_path = os.path.join(settings.VECTOR_STORE_DIR, "faiss_index")
#         self.chunks_path = os.path.join(settings.VECTOR_STORE_DIR, "chunks.pkl")
#         self._initialize_index()
    
#     def _initialize_index(self):
#         """Initialize or load FAISS index"""
#         try:
#             # Try to load existing index
#             if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
#                 self._load_index()
#             else:
#                 # Create new index
#                 self.index = faiss.IndexFlatIP(self.dimension)  # Inner product similarity
#                 self.chunks = []
#                 logger.info("Created new FAISS index")
        
#         except Exception as e:
#             logger.error(f"Failed to initialize vector store: {e}")
#             # Fallback to new index
#             self.index = faiss.IndexFlatIP(self.dimension)
#             self.chunks = []
    
#     def _load_index(self):
#         """Load existing FAISS index and chunks"""
#         try:
#             self.index = faiss.read_index(self.index_path)
#             with open(self.chunks_path, 'rb') as f:
#                 self.chunks = pickle.load(f)
#             logger.info(f"Loaded existing index with {len(self.chunks)} chunks")
        
#         except Exception as e:
#             logger.error(f"Failed to load index: {e}")
#             raise e
    
#     def _save_index(self):
#         """Save FAISS index and chunks to disk"""
#         try:
#             os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)
#             faiss.write_index(self.index, self.index_path)
#             with open(self.chunks_path, 'wb') as f:
#                 pickle.dump(self.chunks, f)
#             logger.info("Saved vector store to disk")
        
#         except Exception as e:
#             logger.error(f"Failed to save index: {e}")
    
#     def _text_similarity(self, text1: str, text2: str) -> float:
#         """Calculate text similarity using sequence matcher"""
#         return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
#     def _deduplicate_chunks(self, chunks_with_scores: List[Tuple[DocumentChunk, float]], 
#                            similarity_threshold: float = 0.7) -> List[Tuple[DocumentChunk, float]]:
#         """Remove duplicate and highly similar chunks"""
#         deduplicated = []
#         seen_chunk_ids = set()
        
#         # Sort by relevance score (highest first)
#         chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        
#         for chunk, score in chunks_with_scores:
#             # Skip if we've already seen this exact chunk
#             if chunk.id in seen_chunk_ids:
#                 continue
            
#             # Check for text similarity with already selected chunks
#             is_duplicate = False
#             for existing_chunk, _ in deduplicated:
#                 text_sim = self._text_similarity(chunk.text, existing_chunk.text)
                
#                 # Skip if too similar to an existing chunk
#                 if text_sim > similarity_threshold:
#                     # Keep the one with higher score
#                     if score > _:  # Current score is higher
#                         # Remove the existing one and add current
#                         deduplicated = [item for item in deduplicated if item[0].id != existing_chunk.id]
#                         seen_chunk_ids.discard(existing_chunk.id)
#                         break
#                     else:
#                         # Keep existing, skip current
#                         is_duplicate = True
#                         break
            
#             if not is_duplicate:
#                 deduplicated.append((chunk, score))
#                 seen_chunk_ids.add(chunk.id)
        
#         return deduplicated
    
#     def _diversify_sources(self, chunks_with_scores: List[Tuple[DocumentChunk, float]], 
#                           max_per_document: int = 2) -> List[Tuple[DocumentChunk, float]]:
#         """Limit chunks per document to increase diversity"""
#         document_counts = {}
#         diversified = []
        
#         for chunk, score in chunks_with_scores:
#             doc_id = chunk.metadata.get('document_id', 'unknown')
            
#             # Count chunks per document
#             if doc_id not in document_counts:
#                 document_counts[doc_id] = 0
            
#             # Add if under limit
#             if document_counts[doc_id] < max_per_document:
#                 diversified.append((chunk, score))
#                 document_counts[doc_id] += 1
        
#         return diversified

#     async def add_chunks(self, chunks: List[DocumentChunk], embedding_service):
#         """Add document chunks to vector store"""
#         try:
#             if not chunks:
#                 return
            
#             # Generate embeddings for all chunks
#             texts = [chunk.text for chunk in chunks]
#             embeddings = embedding_service.generate_embeddings(texts)
            
#             # Normalize embeddings for cosine similarity
#             embeddings_array = np.array(embeddings).astype('float32')
#             faiss.normalize_L2(embeddings_array)
            
#             # Add to FAISS index
#             self.index.add(embeddings_array)
            
#             # Store chunks with embeddings
#             for chunk, embedding in zip(chunks, embeddings):
#                 chunk.embedding = embedding
#                 self.chunks.append(chunk)
            
#             # Save to disk
#             self._save_index()
            
#             logger.info(f"Added {len(chunks)} chunks to vector store")
        
#         except Exception as e:
#             logger.error(f"Failed to add chunks to vector store: {e}")
#             raise e
    
#     async def search_similar(self, query: str, embedding_service, k: int = 10) -> List[Source]:
#         """Search for similar chunks with improved deduplication"""
#         try:
#             if self.index.ntotal == 0:
#                 return []
            
#             # Generate query embedding
#             query_embedding = embedding_service.generate_single_embedding(query)
#             query_vector = np.array([query_embedding]).astype('float32')
#             faiss.normalize_L2(query_vector)
            
#             # Search for more results initially to allow for deduplication
#             search_k = min(k * 3, self.index.ntotal)  # Get 3x more results
#             scores, indices = self.index.search(query_vector, search_k)
            
#             # Convert results to chunks with scores
#             chunks_with_scores = []
#             for score, idx in zip(scores[0], indices[0]):
#                 if idx < len(self.chunks) and score > 0.2:  # Lower threshold for initial filtering
#                     chunk = self.chunks[idx]
#                     chunks_with_scores.append((chunk, float(score)))
            
#             # Apply deduplication
#             deduplicated_chunks = self._deduplicate_chunks(chunks_with_scores, similarity_threshold=0.7)
            
#             # Apply diversity filter
#             diversified_chunks = self._diversify_sources(deduplicated_chunks, max_per_document=2)
            
#             # Take top k results after deduplication
#             final_chunks = diversified_chunks[:k]
            
#             # Convert to sources
#             sources = []
#             for chunk, score in final_chunks:
#                 source = Source(
#                     chunk_id=chunk.id,
#                     text=chunk.text,
#                     metadata=chunk.metadata,
#                     relevance_score=score
#                 )
#                 sources.append(source)
            
#             logger.info(f"Found {len(sources)} unique relevant sources for query")
#             return sources
        
#         except Exception as e:
#             logger.error(f"Failed to search vector store: {e}")
#             return []





import os
import pickle
import json
import numpy as np
import faiss
from typing import List, Tuple, Set, Dict, Any
from loguru import logger
from difflib import SequenceMatcher
from datetime import datetime

from app.core.config import settings
from app.models.schemas import DocumentChunk, Source

class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []
        self.documents_metadata = {}  # NEW: Track document metadata
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        self.index_path = os.path.join(settings.VECTOR_STORE_DIR, "faiss_index")
        self.chunks_path = os.path.join(settings.VECTOR_STORE_DIR, "chunks.pkl")
        self.documents_path = os.path.join(settings.VECTOR_STORE_DIR, "documents_metadata.json")  # NEW
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or load FAISS index"""
        try:
            # Try to load existing index
            if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
                self._load_index()
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product similarity
                self.chunks = []
                self.documents_metadata = {}
                logger.info("Created new FAISS index")
        
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            # Fallback to new index
            self.index = faiss.IndexFlatIP(self.dimension)
            self.chunks = []
            self.documents_metadata = {}
    
    def _load_index(self):
        """Load existing FAISS index and chunks"""
        try:
            self.index = faiss.read_index(self.index_path)
            with open(self.chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            # Load documents metadata if exists
            if os.path.exists(self.documents_path):
                with open(self.documents_path, 'r') as f:
                    self.documents_metadata = json.load(f)
            else:
                self.documents_metadata = {}
                
            logger.info(f"Loaded existing index with {len(self.chunks)} chunks and {len(self.documents_metadata)} documents")
        
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise e
    
    def _save_index(self):
        """Save FAISS index, chunks, and documents metadata to disk"""
        try:
            os.makedirs(settings.VECTOR_STORE_DIR, exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            # Save documents metadata
            with open(self.documents_path, 'w') as f:
                json.dump(self.documents_metadata, f, indent=2, default=str)
                
            logger.info("Saved vector store to disk")
        
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using sequence matcher"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
    
    def _deduplicate_chunks(self, chunks_with_scores: List[Tuple[DocumentChunk, float]], 
                           similarity_threshold: float = 0.7) -> List[Tuple[DocumentChunk, float]]:
        """Remove duplicate and highly similar chunks"""
        deduplicated = []
        seen_chunk_ids = set()
        
        # Sort by relevance score (highest first)
        chunks_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        for chunk, score in chunks_with_scores:
            # Skip if we've already seen this exact chunk
            if chunk.id in seen_chunk_ids:
                continue
            
            # Check for text similarity with already selected chunks
            is_duplicate = False
            for existing_chunk, existing_score in deduplicated:
                text_sim = self._text_similarity(chunk.text, existing_chunk.text)
                
                # Skip if too similar to an existing chunk
                if text_sim > similarity_threshold:
                    # Keep the one with higher score
                    if score > existing_score:  # Current score is higher
                        # Remove the existing one and add current
                        deduplicated = [item for item in deduplicated if item[0].id != existing_chunk.id]
                        seen_chunk_ids.discard(existing_chunk.id)
                        break
                    else:
                        # Keep existing, skip current
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                deduplicated.append((chunk, score))
                seen_chunk_ids.add(chunk.id)
        
        return deduplicated
    
    def _diversify_sources(self, chunks_with_scores: List[Tuple[DocumentChunk, float]], 
                          max_per_document: int = 2) -> List[Tuple[DocumentChunk, float]]:
        """Limit chunks per document to increase diversity"""
        document_counts = {}
        diversified = []
        
        for chunk, score in chunks_with_scores:
            doc_id = chunk.metadata.get('document_id', 'unknown')
            
            # Count chunks per document
            if doc_id not in document_counts:
                document_counts[doc_id] = 0
            
            # Add if under limit
            if document_counts[doc_id] < max_per_document:
                diversified.append((chunk, score))
                document_counts[doc_id] += 1
        
        return diversified

    async def add_chunks(self, chunks: List[DocumentChunk], embedding_service, document_info: Dict[str, Any] = None):
        """Add document chunks to vector store with document tracking"""
        try:
            if not chunks:
                return
            
            # Generate embeddings for all chunks
            texts = [chunk.text for chunk in chunks]
            embeddings = embedding_service.generate_embeddings(texts)
            
            # Normalize embeddings for cosine similarity
            embeddings_array = np.array(embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Store chunks with embeddings
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding
                self.chunks.append(chunk)
            
            # NEW: Store document metadata for frontend retrieval
            if document_info and chunks:
                doc_id = chunks[0].metadata.get('document_id')
                if doc_id:
                    self.documents_metadata[doc_id] = {
                        'document_id': doc_id,
                        'filename': document_info.get('filename', 'Unknown'),
                        'chunks_count': len(chunks),
                        'upload_time': datetime.now().isoformat(),
                        'file_size': document_info.get('file_size', 0),
                        'status': 'processed'
                    }
            
            # Save to disk
            self._save_index()
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
        
        except Exception as e:
            logger.error(f"Failed to add chunks to vector store: {e}")
            raise e
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all uploaded documents metadata for frontend display"""
        try:
            documents_list = list(self.documents_metadata.values())
            # Sort by upload time (newest first)
            documents_list.sort(key=lambda x: x.get('upload_time', ''), reverse=True)
            return documents_list
        except Exception as e:
            logger.error(f"Failed to get documents list: {e}")
            return []
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents"""
        try:
            total_documents = len(self.documents_metadata)
            total_chunks = len(self.chunks)
            
            # Calculate average chunks per document
            avg_chunks = total_chunks / max(total_documents, 1)
            
            return {
                'total_documents': total_documents,
                'total_chunks': total_chunks,
                'average_chunks_per_document': round(avg_chunks, 1)
            }
        except Exception as e:
            logger.error(f"Failed to get document stats: {e}")
            return {'total_documents': 0, 'total_chunks': 0, 'average_chunks_per_document': 0}

    async def search_similar(self, query: str, embedding_service, k: int = 10) -> List[Source]:
        """Search for similar chunks with improved deduplication"""
        try:
            if self.index.ntotal == 0:
                return []
            
            # Generate query embedding
            query_embedding = embedding_service.generate_single_embedding(query)
            query_vector = np.array([query_embedding]).astype('float32')
            faiss.normalize_L2(query_vector)
            
            # Search for more results initially to allow for deduplication
            search_k = min(k * 3, self.index.ntotal)  # Get 3x more results
            scores, indices = self.index.search(query_vector, search_k)
            
            # Convert results to chunks with scores
            chunks_with_scores = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks) and score > 0.2:  # Lower threshold for initial filtering
                    chunk = self.chunks[idx]
                    chunks_with_scores.append((chunk, float(score)))
            
            # Apply deduplication
            deduplicated_chunks = self._deduplicate_chunks(chunks_with_scores, similarity_threshold=0.7)
            
            # Apply diversity filter
            diversified_chunks = self._diversify_sources(deduplicated_chunks, max_per_document=2)
            
            # Take top k results after deduplication
            final_chunks = diversified_chunks[:k]
            
            # Convert to sources
            sources = []
            for chunk, score in final_chunks:
                source = Source(
                    chunk_id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    relevance_score=score
                )
                sources.append(source)
            
            logger.info(f"Found {len(sources)} unique relevant sources for query")
            return sources
        
        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            return []