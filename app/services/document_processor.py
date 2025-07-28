# import os
# import uuid
# from typing import List, Tuple
# from pathlib import Path
# import PyPDF2
# from loguru import logger

# from app.core.config import settings
# from app.models.schemas import DocumentChunk

# class DocumentProcessor:
#     def __init__(self):
#         self.chunk_size = settings.CHUNK_SIZE
#         self.chunk_overlap = settings.CHUNK_OVERLAP
    
#     async def process_file(self, file_path: str, filename: str) -> Tuple[List[DocumentChunk], str]:
#         """Process uploaded file and return chunks"""
#         try:
#             # Generate document ID
#             doc_id = str(uuid.uuid4())
            
#             # Extract text based on file type
#             file_extension = Path(filename).suffix.lower()
            
#             if file_extension == '.pdf':
#                 text = self._extract_pdf_text(file_path)
#             elif file_extension == '.txt':
#                 text = self._extract_txt_text(file_path)
#             else:
#                 raise ValueError(f"Unsupported file type: {file_extension}")
            
#             # Create chunks
#             chunks = self._create_chunks(text, doc_id, filename)
            
#             logger.info(f"Processed document {filename}: {len(chunks)} chunks created")
#             return chunks, doc_id
        
#         except Exception as e:
#             logger.error(f"Failed to process file {filename}: {e}")
#             raise e
    
#     def _extract_pdf_text(self, file_path: str) -> str:
#         """Extract text from PDF file"""
#         try:
#             with open(file_path, 'rb') as file:
#                 pdf_reader = PyPDF2.PdfReader(file)
#                 text = ""
                
#                 for page_num, page in enumerate(pdf_reader.pages):
#                     page_text = page.extract_text()
#                     text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
#                 return text.strip()
        
#         except Exception as e:
#             logger.error(f"Failed to extract PDF text: {e}")
#             raise e
    
#     def _extract_txt_text(self, file_path: str) -> str:
#         """Extract text from TXT file"""
#         try:
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 return file.read().strip()
        
#         except Exception as e:
#             logger.error(f"Failed to extract TXT text: {e}")
#             raise e
    
#     def _create_chunks(self, text: str, doc_id: str, filename: str) -> List[DocumentChunk]:
#         """Create overlapping chunks from text"""
#         chunks = []
#         start = 0
#         chunk_index = 0
        
#         while start < len(text):
#             # Calculate end position
#             end = start + self.chunk_size
            
#             # Get chunk text
#             chunk_text = text[start:end]
            
#             # Skip very small chunks
#             if len(chunk_text.strip()) < 50:
#                 break
            
#             # Create chunk
#             chunk = DocumentChunk(
#                 id=f"{doc_id}_{chunk_index}",
#                 text=chunk_text.strip(),
#                 metadata={
#                     "document_id": doc_id,
#                     "filename": filename,
#                     "chunk_index": chunk_index,
#                     "start_char": start,
#                     "end_char": end
#                 }
#             )
            
#             chunks.append(chunk)
            
#             # Move to next chunk with overlap
#             start += self.chunk_size - self.chunk_overlap
#             chunk_index += 1
        
#         return chunks


# app/services/document_processor.py
import os
import uuid
import random
from typing import List, Tuple
from pathlib import Path
import PyPDF2
from loguru import logger
import google.generativeai as genai

from app.core.config import settings
from app.models.schemas import DocumentChunk

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        # Initialize Gemini for validation
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.validation_model = genai.GenerativeModel('gemini-1.5-flash')
    
    def _validate_insurance_content(self, text: str) -> Tuple[bool, str]:
        """
        Validate if document content is insurance-related using LLM
        Takes random chunks and asks LLM to determine if it's insurance content
        """
        try:
            # Split text into words for chunking
            words = text.split()
            
            # If text is too short, use the whole text
            if len(words) < 500:
                sample_text = text
            else:
                # Create 2-3 random chunks of 500-700 words each
                chunks = []
                num_samples = min(3, len(words) // 500)  # 2-3 chunks
                
                for _ in range(num_samples):
                    # Random starting position
                    start_pos = random.randint(0, max(0, len(words) - 700))
                    end_pos = min(start_pos + random.randint(500, 700), len(words))
                    
                    chunk = ' '.join(words[start_pos:end_pos])
                    chunks.append(chunk)
                
                # Combine random chunks
                sample_text = '\n\n---SAMPLE CHUNK---\n\n'.join(chunks)
            
            # Create validation prompt
            validation_prompt = f"""You are tasked with determining if the following document content is related to insurance policies, coverage, benefits, or insurance industry.

DOCUMENT SAMPLE:
{sample_text}

Analyze the content and determine:
1. Is this content related to insurance, health coverage, policy terms, benefits, claims, premiums, or insurance industry?
2. Does it contain insurance terminology like: policy, coverage, premium, deductible, claim, benefit, insured, copay, etc.?

Respond with ONLY:
- "YES" if this appears to be insurance-related content
- "NO" if this appears to be non-insurance content (like resume, recipe, manual, news, etc.)

Answer:"""

            # Make LLM call for validation
            response = self.validation_model.generate_content(validation_prompt)
            
            if response.text:
                answer = response.text.strip().upper()
                if "YES" in answer:
                    return True, "Document validated as insurance-related content by AI analysis"
                else:
                    return False, "Document does not appear to be insurance-related content"
            else:
                return False, "Unable to validate document content"
                
        except Exception as e:
            logger.error(f"Error validating document with LLM: {e}")
            return False, f"Error during document validation: {str(e)}"
    
    async def process_file(self, file_path: str, filename: str) -> Tuple[List[DocumentChunk], str]:
        """Process uploaded file and return chunks with insurance validation"""
        try:
            # Generate document ID
            doc_id = str(uuid.uuid4())
            
            # Extract text based on file type
            file_extension = Path(filename).suffix.lower()
            
            if file_extension == '.pdf':
                text = self._extract_pdf_text(file_path)
            elif file_extension == '.txt':
                text = self._extract_txt_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # INSURANCE VALIDATION - NEW ADDITION
            logger.info(f"Validating document {filename} for insurance content...")
            is_valid, reason = self._validate_insurance_content(text)
            
            if not is_valid:
                logger.warning(f"Document {filename} failed insurance validation: {reason}")
                raise ValueError(f"This document does not appear to be insurance-related. Please upload insurance policy documents, certificates of coverage, or benefits summaries only.")
            
            logger.info(f"Document {filename} passed insurance validation: {reason}")
            
            # Create chunks
            chunks = self._create_chunks(text, doc_id, filename)
            
            logger.info(f"Processed insurance document {filename}: {len(chunks)} chunks created")
            return chunks, doc_id
        
        except Exception as e:
            logger.error(f"Failed to process file {filename}: {e}")
            raise e
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                return text.strip()
        
        except Exception as e:
            logger.error(f"Failed to extract PDF text: {e}")
            raise e
    
    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        
        except Exception as e:
            logger.error(f"Failed to extract TXT text: {e}")
            raise e
    
    def _create_chunks(self, text: str, doc_id: str, filename: str) -> List[DocumentChunk]:
        """Create overlapping chunks from text"""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Get chunk text
            chunk_text = text[start:end]
            
            # Skip very small chunks
            if len(chunk_text.strip()) < 50:
                break
            
            # Create chunk
            chunk = DocumentChunk(
                id=f"{doc_id}_{chunk_index}",
                text=chunk_text.strip(),
                metadata={
                    "document_id": doc_id,
                    "filename": filename,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    "document_type": "insurance_validated"  # Mark as validated insurance content
                }
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start += self.chunk_size - self.chunk_overlap
            chunk_index += 1
        
        return chunks