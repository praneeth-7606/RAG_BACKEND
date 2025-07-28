# app/core/insurance_validator.py
import re
from typing import Tuple, List
from loguru import logger

class InsuranceValidator:
    """Validates content and queries to ensure they are insurance-related"""
    
    def __init__(self):
        # Insurance-related keywords and terms
        self.insurance_keywords = {
            'policy_terms': [
                'policy', 'coverage', 'premium', 'deductible', 'claim', 'benefit',
                'liability', 'insured', 'insurer', 'policyholder', 'beneficiary',
                'exclusion', 'copay', 'coinsurance', 'out-of-pocket', 'network'
            ],
            'insurance_types': [
                'health insurance', 'life insurance', 'auto insurance', 'home insurance',
                'property insurance', 'casualty insurance', 'disability insurance',
                'umbrella insurance', 'renters insurance', 'business insurance',
                'workers compensation', 'professional liability', 'malpractice'
            ],
            'medical_terms': [
                'medical', 'hospital', 'prescription', 'doctor', 'physician',
                'treatment', 'diagnosis', 'procedure', 'surgery', 'medication',
                'preventive care', 'emergency', 'specialist', 'therapy'
            ],
            'financial_terms': [
                'reimbursement', 'payment', 'cost', 'expense', 'fee', 'charge',
                'allowable', 'maximum', 'limit', 'annual', 'monthly', 'quarterly'
            ]
        }
        
        # Non-insurance content indicators
        self.non_insurance_indicators = [
            'resume', 'cv', 'curriculum vitae', 'work experience', 'education',
            'skills', 'employment history', 'job description', 'salary',
            'recipe', 'cooking', 'ingredients', 'instructions',
            'news article', 'sports', 'entertainment', 'politics',
            'academic paper', 'research', 'thesis', 'dissertation',
            'manual', 'user guide', 'technical documentation'
        ]
        
        # Insurance question patterns
        self.insurance_question_patterns = [
            r'what.*coverage',
            r'how.*claim',
            r'what.*premium',
            r'what.*deductible',
            r'what.*benefit',
            r'what.*policy',
            r'how.*copay',
            r'what.*excluded',
            r'what.*covered',
            r'how.*reimburs',
            r'what.*network',
            r'what.*provider',
            r'what.*limit'
        ]
    
    def validate_document_content(self, text: str, filename: str) -> Tuple[bool, str, float]:
        """
        Validate if document content is insurance-related
        Returns: (is_valid, reason, confidence_score)
        """
        try:
            text_lower = text.lower()
            
            # Check for non-insurance indicators first
            non_insurance_score = 0
            for indicator in self.non_insurance_indicators:
                if indicator in text_lower:
                    non_insurance_score += 1
            
            # If too many non-insurance indicators, reject
            if non_insurance_score >= 3:
                return False, f"Document appears to be a {', '.join([ind for ind in self.non_insurance_indicators if ind in text_lower][:3])} rather than an insurance document", 0.1
            
            # Calculate insurance relevance score
            insurance_score = 0
            total_keywords = 0
            
            for category, keywords in self.insurance_keywords.items():
                category_score = 0
                for keyword in keywords:
                    count = text_lower.count(keyword)
                    category_score += count
                    total_keywords += len(keywords)
                
                insurance_score += category_score
            
            # Calculate confidence as percentage
            confidence = min(insurance_score / max(len(text.split()) * 0.01, 1), 1.0)
            
            # Minimum threshold for insurance content
            min_threshold = 0.05  # At least 5% insurance-related content
            
            if confidence >= min_threshold:
                return True, f"Document validated as insurance-related (confidence: {confidence:.2%})", confidence
            else:
                return False, f"Document does not appear to contain sufficient insurance-related content (confidence: {confidence:.2%})", confidence
                
        except Exception as e:
            logger.error(f"Error validating document content: {e}")
            return False, "Error validating document content", 0.0
    
    def validate_query(self, query: str) -> Tuple[bool, str]:
        """
        Validate if query is insurance-related
        Returns: (is_valid, reason)
        """
        try:
            query_lower = query.lower()
            
            # Check for insurance keywords in query
            insurance_keyword_found = False
            for category, keywords in self.insurance_keywords.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        insurance_keyword_found = True
                        break
                if insurance_keyword_found:
                    break
            
            # Check for insurance question patterns
            pattern_match = False
            for pattern in self.insurance_question_patterns:
                if re.search(pattern, query_lower):
                    pattern_match = True
                    break
            
            # Accept if either keywords found or pattern matches
            if insurance_keyword_found or pattern_match:
                return True, "Query is insurance-related"
            
            # Check for obvious non-insurance queries
            non_insurance_terms = [
                'recipe', 'cooking', 'weather', 'sports', 'news', 'entertainment',
                'job', 'career', 'salary', 'resume', 'education', 'travel',
                'shopping', 'fashion', 'technology', 'programming', 'gaming'
            ]
            
            for term in non_insurance_terms:
                if term in query_lower:
                    return False, f"Query appears to be about {term}, not insurance"
            
            # If no clear insurance indicators but no clear non-insurance terms either
            # Be more permissive and ask for clarification
            return False, "Please ask questions specifically about insurance policies, coverage, claims, or benefits"
            
        except Exception as e:
            logger.error(f"Error validating query: {e}")
            return False, "Error processing your question"
    
    def get_insurance_context_prompt(self) -> str:
        """Get enhanced prompt for insurance-only responses"""
        return """You are a specialized Insurance Policy AI Assistant. You ONLY answer questions about insurance policies, coverage, claims, benefits, and related insurance topics.

STRICT GUIDELINES:
1. ONLY answer questions about insurance, health coverage, policy terms, claims, benefits, premiums, deductibles, etc.
2. If asked about non-insurance topics (jobs, recipes, general knowledge, etc.), politely decline and redirect to insurance topics
3. If the uploaded document doesn't appear to be insurance-related, inform the user that you only work with insurance documents
4. Always base your answers on the provided insurance document context
5. Use professional insurance terminology
6. Be specific about coverage details, limits, and exclusions when available

REFUSE to answer questions about:
- Personal advice unrelated to insurance
- General knowledge questions
- Career, job, or resume advice
- Recipes, cooking, or lifestyle topics
- Technical support for non-insurance products
- Any topic outside of insurance and policy coverage"""