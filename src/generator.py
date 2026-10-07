"""
Answer generation module for DocuMind.
Handles LLM API calls to Gemini, Groq, and OpenAI with retry logic for rate limits.
"""

import os
import time
import logging
import importlib
from typing import Dict, List, Optional

from google import genai
from google.genai import types
from groq import Groq
from openai import OpenAI

from config import (
    LLM_BACKEND, GOOGLE_API_KEY, GROQ_API_KEY, OPENAI_API_KEY,
    GEMINI_MODEL, GROQ_MODEL, OPENAI_MODEL,
    GEMINI_TIMEOUT, GROQ_TIMEOUT, OPENAI_TIMEOUT,
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_package_installed(package_name: str) -> bool:
    """
    Check if a Python package is installed.
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        True if installed, False otherwise
    """
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False


def print_configuration_diagnostics():
    """Print configuration diagnostics without exposing secrets."""
    logger.info("=" * 60)
    logger.info("[CONFIGURATION DIAGNOSTICS]")
    logger.info("=" * 60)
    
    # Check package installation
    gemini_installed = check_package_installed("google.genai")
    groq_installed = check_package_installed("groq")
    openai_installed = check_package_installed("openai")
    
    logger.info(f"Gemini package installed: {gemini_installed}")
    logger.info(f"Groq package installed: {groq_installed}")
    logger.info(f"OpenAI package installed: {openai_installed}")
    
    # Check API keys (without exposing values)
    gemini_key_configured = bool(GOOGLE_API_KEY and GOOGLE_API_KEY.strip())
    groq_key_configured = bool(GROQ_API_KEY and GROQ_API_KEY.strip())
    openai_key_configured = bool(OPENAI_API_KEY and OPENAI_API_KEY.strip())
    
    logger.info(f"Gemini API key: {'configured' if gemini_key_configured else 'NOT configured'}")
    logger.info(f"Groq API key: {'configured' if groq_key_configured else 'NOT configured'}")
    logger.info(f"OpenAI API key: {'configured' if openai_key_configured else 'NOT configured'}")
    
    # Model names
    logger.info(f"Gemini model: {GEMINI_MODEL}")
    logger.info(f"Groq model: {GROQ_MODEL}")
    logger.info(f"OpenAI model: {OPENAI_MODEL}")
    
    # Selected backend
    logger.info(f"Selected backend: {LLM_BACKEND}")
    
    # Timeout settings
    logger.info(f"Gemini timeout: {GEMINI_TIMEOUT}s")
    logger.info(f"Groq timeout: {GROQ_TIMEOUT}s")
    logger.info(f"OpenAI timeout: {OPENAI_TIMEOUT}s")
    
    logger.info("=" * 60)


class LLMGenerator:
    """Generate answers using LLM APIs with automatic backend selection."""
    
    def __init__(self):
        self.backend = LLM_BACKEND
        self.gemini_client = None
        self.gemini_model_name = None
        self.groq_client = None
        self.openai_client = None
        
        # Print configuration diagnostics on startup
        print_configuration_diagnostics()
        
        self._setup_gemini()
        self._setup_groq()
        self._setup_openai()
    
    def _setup_gemini(self):
        """Initialize Gemini API client."""
        if GOOGLE_API_KEY:
            try:
                self.gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
                # Store the model name for use in generate_content
                self.gemini_model_name = GEMINI_MODEL
                logger.info(f"[LLM] Gemini client initialized with model: {GEMINI_MODEL}")
            except Exception as e:
                logger.error(f"[LLM] Error initializing Gemini: {str(e)}")
                self.gemini_client = None
                self.gemini_model_name = None
        else:
            logger.warning("[LLM] GOOGLE_API_KEY not set. Gemini backend unavailable.")
    
    def _setup_groq(self):
        """Initialize Groq API client."""
        if GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=GROQ_API_KEY)
                logger.info(f"[LLM] Groq client initialized with model: {GROQ_MODEL}")
            except Exception as e:
                logger.error(f"[LLM] Error initializing Groq: {str(e)}")
                self.groq_client = None
        else:
            logger.warning("[LLM] GROQ_API_KEY not set. Groq backend unavailable.")
    
    def _setup_openai(self):
        """Initialize OpenAI API client."""
        if OPENAI_API_KEY:
            try:
                self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
                logger.info(f"[LLM] OpenAI client initialized with model: {OPENAI_MODEL}")
            except Exception as e:
                logger.error(f"[LLM] Error initializing OpenAI: {str(e)}")
                self.openai_client = None
        else:
            logger.warning("[LLM] OPENAI_API_KEY not set. OpenAI backend unavailable.")
    
    def _get_client_for_backend(self, backend: str):
        """Return the initialized client object for a given backend name, or None."""
        if backend == "gemini":
            return self.gemini_client
        elif backend == "groq":
            return self.groq_client
        elif backend == "openai":
            return self.openai_client
        return None
    
    def _generate_for_backend(self, backend: str, prompt: str) -> Dict[str, str]:
        """Dispatch generation to the correct backend-specific method."""
        if backend == "gemini":
            return self._generate_with_gemini(prompt)
        elif backend == "groq":
            return self._generate_with_groq(prompt)
        elif backend == "openai":
            return self._generate_with_openai(prompt)
        else:
            raise ValueError(f"Unknown LLM backend: {backend}")
    
    def generate_answer(self, context: str, question: str) -> Dict[str, str]:
        """
        Generate an answer to the question using the retrieved context.
        
        Args:
            context: Retrieved context string
            question: User question
            
        Returns:
            Dictionary with 'answer' and 'backend' keys
        """
        # Build the prompt
        prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)
        
        # Try the configured backend first
        primary_backend = self.backend
        
        # Build fallback order: try the other two backends in a fixed order,
        # skipping whichever one is already the primary
        all_backends = ["gemini", "groq", "openai"]
        fallback_order = [b for b in all_backends if b != primary_backend]
        
        logger.info(f"[LLM] Trying {primary_backend}...")
        
        try:
            return self._generate_for_backend(primary_backend, prompt)
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[LLM] {primary_backend} failed: {error_msg}")
            
            errors_log = {primary_backend: error_msg}
            
            # Try each fallback backend in order, skipping ones without a configured client
            for fallback_backend in fallback_order:
                if self._get_client_for_backend(fallback_backend) is None:
                    continue
                
                logger.info(f"[LLM] Falling back to {fallback_backend}...")
                try:
                    return self._generate_for_backend(fallback_backend, prompt)
                except Exception as fallback_error:
                    fallback_error_msg = str(fallback_error)
                    logger.error(f"[LLM] {fallback_backend} fallback failed: {fallback_error_msg}")
                    errors_log[fallback_backend] = fallback_error_msg
            
            # If we reach here, every available backend failed (or none were configured)
            errors_summary = ", ".join(f"{b} error: {msg}" for b, msg in errors_log.items())
            raise Exception(f"All available LLM backends failed. {errors_summary}")
    
    def _is_retryable_error(self, error_msg: str) -> bool:
        """
        Check if an error is retryable.
        
        Args:
            error_msg: The error message to check
            
        Returns:
            True if retryable, False otherwise
        """
        # Retry on timeouts and rate limits
        if "504" in error_msg or "Deadline expired" in error_msg:
            return True
        if "503" in error_msg or "Service Unavailable" in error_msg:
            return True
        if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
            return True
        if "timeout" in error_msg.lower():
            return True
        # Don't retry on configuration errors
        if "not set" in error_msg or "not initialized" in error_msg:
            return False
        # Don't retry on model_not_found errors (404)
        if "404" in error_msg or "not found" in error_msg.lower() or "model_not_found" in error_msg.lower():
            return False
        # Don't retry on authentication errors
        if "401" in error_msg or "403" in error_msg or "authentication" in error_msg.lower() or "permission" in error_msg.lower():
            return False
        # Retry on other API errors (network issues, etc.)
        return True
    
    def _generate_with_gemini(self, prompt: str) -> Dict[str, str]:
        """
        Generate answer using Gemini API with retry logic.
        
        Args:
            prompt: Complete prompt including context and question
            
        Returns:
            Dictionary with 'answer' and 'backend' keys
        """
        if not GOOGLE_API_KEY:
            raise Exception("GOOGLE_API_KEY not set")
        
        if self.gemini_client is None:
            raise Exception("Gemini client not initialized. Check API key.")
        
        # Manual retry logic
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Combine system prompt with user prompt
                full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
                
                # Generate response using new API
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        max_output_tokens=1024,
                    )
                )
                
                answer = response.text
                
                logger.info(f"[LLM] Gemini response generated successfully")
                
                return {
                    'answer': answer,
                    'backend': 'gemini'
                }
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if error is retryable
                if not self._is_retryable_error(error_msg):
                    # Don't retry on non-retryable errors
                    raise Exception(f"Gemini error: {error_msg}")
                
                # If this is the last attempt, raise
                if attempt == max_retries - 1:
                    raise Exception(f"Gemini error after {max_retries} retries: {error_msg}")
                
                # Wait before retry
                delay = base_delay * (2 ** attempt)
                logger.warning(f"[LLM] Gemini attempt {attempt + 1}/{max_retries} failed: {error_msg}. Retrying in {delay}s...")
                time.sleep(delay)
    
    def _generate_with_groq(self, prompt: str) -> Dict[str, str]:
        """
        Generate answer using Groq API with retry logic.
        
        Args:
            prompt: Complete prompt including context and question
            
        Returns:
            Dictionary with 'answer' and 'backend' keys
        """
        if not GROQ_API_KEY:
            raise Exception("GROQ_API_KEY not set")
        
        if self.groq_client is None:
            raise Exception("Groq client not initialized. Check API key.")
        
        # Manual retry logic
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Combine system prompt with user prompt
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
                
                # Generate response with timeout
                response = self.groq_client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                    timeout=GROQ_TIMEOUT
                )
                
                answer = response.choices[0].message.content
                
                logger.info(f"[LLM] Groq response generated successfully")
                
                return {
                    'answer': answer,
                    'backend': 'groq'
                }
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if error is retryable
                if not self._is_retryable_error(error_msg):
                    # Don't retry on non-retryable errors
                    raise Exception(f"Groq error: {error_msg}")
                
                # If this is the last attempt, raise
                if attempt == max_retries - 1:
                    raise Exception(f"Groq error after {max_retries} retries: {error_msg}")
                
                # Wait before retry
                delay = base_delay * (2 ** attempt)
                logger.warning(f"[LLM] Groq attempt {attempt + 1}/{max_retries} failed: {error_msg}. Retrying in {delay}s...")
                time.sleep(delay)
    
    def _generate_with_openai(self, prompt: str) -> Dict[str, str]:
        """
        Generate answer using OpenAI API with retry logic.
        
        Args:
            prompt: Complete prompt including context and question
            
        Returns:
            Dictionary with 'answer' and 'backend' keys
        """
        if not OPENAI_API_KEY:
            raise Exception("OPENAI_API_KEY not set")
        
        if self.openai_client is None:
            raise Exception("OpenAI client not initialized. Check API key.")
        
        # Manual retry logic
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Combine system prompt with user prompt
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
                
                # Generate response with timeout
                response = self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                    timeout=OPENAI_TIMEOUT
                )
                
                answer = response.choices[0].message.content
                
                logger.info(f"[LLM] OpenAI response generated successfully")
                
                return {
                    'answer': answer,
                    'backend': 'openai'
                }
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if error is retryable
                if not self._is_retryable_error(error_msg):
                    # Don't retry on non-retryable errors
                    raise Exception(f"OpenAI error: {error_msg}")
                
                # If this is the last attempt, raise
                if attempt == max_retries - 1:
                    raise Exception(f"OpenAI error after {max_retries} retries: {error_msg}")
                
                # Wait before retry
                delay = base_delay * (2 ** attempt)
                logger.warning(f"[LLM] OpenAI attempt {attempt + 1}/{max_retries} failed: {error_msg}. Retrying in {delay}s...")
                time.sleep(delay)


class AnswerGenerator:
    """Main class for generating answers with citations."""
    
    def __init__(self):
        self.llm_generator = LLMGenerator()
    
    def generate(self, context: str, question: str, retrieval_results: List[Dict]) -> Dict:
        """
        Generate an answer with citations.
        
        Args:
            context: Formatted context string from retriever
            question: User question
            retrieval_results: Original retrieval results for citation extraction
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        # Generate the answer
        try:
            result = self.llm_generator.generate_answer(context, question)
            answer = result['answer']
            backend = result['backend']
            logger.info(f"[LLM] Answer generated using {backend}")
        except Exception as e:
            error_msg = f"Failed to generate answer: {str(e)}"
            logger.error(f"[LLM] {error_msg}")
            return {
                'answer': error_msg,
                'citations': [],
                'backend': 'error',
                'success': False
            }
        
        # Extract citations from retrieval results
        citations = self._extract_citations(retrieval_results)
        
        return {
            'answer': answer,
            'citations': citations,
            'backend': backend,
            'success': True
        }
    
    def _extract_citations(self, retrieval_results: List[Dict]) -> List[Dict]:
        """
        Extract citation information from retrieval results.
        
        Args:
            retrieval_results: List of retrieval results
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        seen = set()  # Avoid duplicate citations
        
        for result in retrieval_results:
            source = result['metadata']['source']
            page = result['metadata']['page']
            citation_key = f"{source}_page_{page}"
            
            if citation_key not in seen:
                citations.append({
                    'source': source,
                    'page': page,
                    'chunk_index': result['metadata']['chunk_index']
                })
                seen.add(citation_key)
        
        return citations


# Main execution for testing
if __name__ == "__main__":
    generator = AnswerGenerator()
    
    # Test with sample context and question
    sample_context = """
    [1] Source: sample.pdf, Page: 1
    This is a sample document about machine learning. Machine learning is a subset of artificial intelligence that enables systems to learn from data.
    
    [2] Source: sample.pdf, Page: 2
    Deep learning is a specialized branch of machine learning that uses neural networks with multiple layers. It has achieved remarkable success in image recognition and natural language processing.
    """
    
    sample_question = "What is machine learning?"
    
    print("Testing answer generation...")
    result = generator.generate(sample_context, sample_question, [])
    
    print(f"\nBackend used: {result['backend']}")
    print(f"Answer: {result['answer']}")
    print(f"Success: {result['success']}")