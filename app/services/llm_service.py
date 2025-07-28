# import google.generativeai as genai
# from typing import List, Dict
# from loguru import logger

# from app.core.config import settings
# from app.models.schemas import Source

# class LLMService:
#     def __init__(self):
#         # Configure Gemini
#         genai.configure(api_key=settings.GEMINI_API_KEY)
#         self.model = genai.GenerativeModel('gemini-1.5-flash')
    
#     def _group_sources_by_document(self, sources: List[Source]) -> Dict[str, List[Source]]:
#         """Group sources by document for better context organization"""
#         grouped = {}
#         for source in sources:
#             doc_name = source.metadata.get('filename', 'Unknown Document')
#             if doc_name not in grouped:
#                 grouped[doc_name] = []
#             grouped[doc_name].append(source)
#         return grouped
    
#     def _build_context(self, sources: List[Source]) -> str:
#         """Build context string from sources with document grouping"""
#         if not sources:
#             return ""
        
#         # Group sources by document
#         grouped_sources = self._group_sources_by_document(sources)
        
#         context_parts = []
#         source_counter = 1
        
#         for doc_name, doc_sources in grouped_sources.items():
#             # Sort sources by relevance within each document
#             doc_sources.sort(key=lambda x: x.relevance_score, reverse=True)
            
#             context_parts.append(f"\n=== From Document: {doc_name} ===")
            
#             for source in doc_sources:
#                 context_parts.append(
#                     f"\nSource {source_counter} (Relevance: {source.relevance_score:.2f}):\n"
#                     f"{source.text}\n"
#                 )
#                 source_counter += 1
        
#         return "\n".join(context_parts)
    
#     def _build_prompt(self, query: str, context: str) -> str:
#         """Build enhanced prompt for the LLM"""
#         prompt = f"""You are an expert insurance policy analyst. Answer the user's question based ONLY on the provided context from insurance documents.

# CONTEXT FROM INSURANCE DOCUMENTS:
# {context}

# USER QUESTION: {query}

# INSTRUCTIONS:
# 1. Answer based EXCLUSIVELY on the provided context
# 2. If the context doesn't contain sufficient information, clearly state this
# 3. Cite specific document sources when making claims
# 4. Be precise about coverage amounts, deductibles, and limits
# 5. Use professional insurance terminology
# 6. If multiple documents provide conflicting information, note the discrepancy

# ANSWER:"""
        
#         return prompt

#     async def generate_answer(self, query: str, sources: List[Source]) -> str:
#         """Generate answer using retrieved sources with improved context"""
#         try:
#             if not sources:
#                 return "I couldn't find relevant information in the uploaded documents to answer your question. Please ensure you've uploaded the appropriate insurance policy documents."
            
#             # Build enhanced context
#             context = self._build_context(sources)
            
#             # Create enhanced prompt
#             prompt = self._build_prompt(query, context)
            
#             # Generate response
#             response = self.model.generate_content(prompt)
            
#             if response.text:
#                 answer = response.text.strip()
                
#                 # Add source summary at the end
#                 doc_names = list(set(source.metadata.get('filename', 'Unknown') 
#                                    for source in sources))
#                 if len(doc_names) == 1:
#                     answer += f"\n\n*Information retrieved from: {doc_names[0]}*"
#                 else:
#                     answer += f"\n\n*Information retrieved from {len(doc_names)} documents: {', '.join(doc_names)}*"
                
#                 return answer
#             else:
#                 return "I couldn't generate a proper response. Please try rephrasing your question or check if the uploaded documents contain relevant information."
        
#         except Exception as e:
#             logger.error(f"Failed to generate answer: {e}")
#             return "I encountered an error while processing your question. Please try again or contact support if the issue persists."




# app/services/llm_service.py
# app/services/llm_service.py
import google.generativeai as genai
from typing import List, Dict
from loguru import logger

from app.core.config import settings
from app.models.schemas import Source

class LLMService:
    def __init__(self):
        # Configure Gemini
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def _group_sources_by_document(self, sources: List[Source]) -> Dict[str, List[Source]]:
        """Group sources by document for better context organization"""
        grouped = {}
        for source in sources:
            doc_name = source.metadata.get('filename', 'Unknown Document')
            if doc_name not in grouped:
                grouped[doc_name] = []
            grouped[doc_name].append(source)
        return grouped
    
    def _build_context(self, sources: List[Source]) -> str:
        """Build context string from sources with document grouping"""
        if not sources:
            return ""
        
        # Group sources by document
        grouped_sources = self._group_sources_by_document(sources)
        
        context_parts = []
        source_counter = 1
        
        for doc_name, doc_sources in grouped_sources.items():
            # Sort sources by relevance within each document
            doc_sources.sort(key=lambda x: x.relevance_score, reverse=True)
            
            context_parts.append(f"\n=== From Document: {doc_name} ===")
            
            for source in doc_sources:
                context_parts.append(
                    f"\nSource {source_counter} (Relevance: {source.relevance_score:.2f}):\n"
                    f"{source.text}\n"
                )
                source_counter += 1
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build enhanced prompt for the LLM with built-in validation"""
        prompt = f"""You are an expert insurance policy analyst. You ONLY answer questions about insurance policies, coverage, benefits, claims, and related insurance topics.

CONTEXT FROM INSURANCE DOCUMENTS:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. FIRST: Determine if this question is about insurance, health coverage, policy terms, benefits, claims, premiums, deductibles, or insurance industry topics.

2. IF THE QUESTION IS NOT ABOUT INSURANCE:
   Respond with: "I can only answer questions related to insurance policies, coverage, benefits, claims, and insurance terms. Please ask a question about your insurance documentation.

   Examples of questions I can help with:
   • What is my deductible?
   • What services are covered?
   • How do I file a claim?
   • What are my copayment amounts?
   • What is excluded from coverage?"

3. IF THE QUESTION IS ABOUT INSURANCE:
   - Answer based EXCLUSIVELY on the provided context
   - If the context doesn't contain sufficient information, clearly state this
   - Cite specific document sources when making claims
   - Be precise about coverage amounts, deductibles, and limits
   - Use professional insurance terminology
   - If multiple documents provide conflicting information, note the discrepancy

ANSWER:"""
        
        return prompt

    async def generate_answer(self, query: str, sources: List[Source]) -> str:
        """Generate answer using retrieved sources with built-in validation"""
        try:
            if not sources:
                return "I couldn't find relevant information in the uploaded documents to answer your question. Please ensure you've uploaded the appropriate insurance policy documents."
            
            # Build enhanced context
            context = self._build_context(sources)
            
            # Create enhanced prompt with built-in validation
            prompt = self._build_prompt(query, context)
            
            # Generate response (single LLM call handles both validation and answering)
            response = self.model.generate_content(prompt)
            
            if response.text:
                answer = response.text.strip()
                
                # Only add source summary if it's not a rejection message
                if not answer.startswith("I can only answer questions"):
                    # Add source summary at the end
                    doc_names = list(set(source.metadata.get('filename', 'Unknown') 
                                       for source in sources))
                    if len(doc_names) == 1:
                        answer += f"\n\n*Information retrieved from: {doc_names[0]}*"
                    else:
                        answer += f"\n\n*Information retrieved from {len(doc_names)} documents: {', '.join(doc_names)}*"
                
                return answer
            else:
                return "I couldn't generate a proper response. Please try rephrasing your question or check if the uploaded documents contain relevant information."
        
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return "I encountered an error while processing your question. Please try again or contact support if the issue persists."