"""
Configuration file for DocuMind RAG chatbot.
All configurable parameters are centralized here for easy modification.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()


# =============================================================================
# LLM BACKEND CONFIGURATION
# =============================================================================
# Choose which LLM backend to use: "gemini", "groq", or "openai"
# Gemini is the primary (free tier), Groq is the backup (free tier), 
# OpenAI is optional (paid, higher quality)
LLM_BACKEND = os.getenv("LLM_BACKEND", "gemini")

# API Keys (set as environment variables)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model names (can be overridden by environment variables)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# API timeout configuration (in seconds)
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "60"))
GROQ_TIMEOUT = int(os.getenv("GROQ_TIMEOUT", "60"))
OPENAI_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "60"))


# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================
# Fixed-size chunking parameters
CHUNK_SIZE = 500  # Number of characters per chunk
CHUNK_OVERLAP = 100  # Number of characters to overlap between chunks

# Sentence-based chunking parameters
SENTENCE_CHUNK_MAX_SIZE = 800  # Max characters per sentence chunk
SENTENCE_CHUNK_OVERLAP = 2  # Number of sentences to overlap


# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================
# Embedding model (running locally, no API needed)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"  # Use "cuda" if GPU is available


# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================
# Number of chunks to retrieve
TOP_K = 5  # Number of top results to retrieve from vector search
TOP_K_BM25 = 10  # Number of results to retrieve from BM25 (for hybrid)

# Hybrid retrieval weights
VECTOR_WEIGHT = 0.7  # Weight for vector similarity scores
BM25_WEIGHT = 0.3  # Weight for BM25 scores


# =============================================================================
# VECTOR STORE CONFIGURATION
# =============================================================================
# ChromaDB persistence
CHROMA_PERSIST_DIR = "vectorstore"
COLLECTION_NAME = "documind_documents"


# =============================================================================
# PDF PROCESSING CONFIGURATION
# =============================================================================
# OCR fallback settings
USE_OCR_FALLBACK = True  # Enable OCR for scanned PDFs
OCR_DPI = 300  # DPI for PDF to image conversion


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================
# Evaluation test data path
EVAL_QA_PATH = "data/eval_qa.json"

# Evaluation metrics
EVAL_TOP_K_VALUES = [3, 5, 10]  # k values for recall@k calculation

# Rate limiting for evaluation (to avoid hitting API limits)
EVAL_DELAY_SECONDS = 2  # Delay between API calls during evaluation


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================
# System prompt for the LLM
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided document context.
Your task is to:
1. Read the provided context carefully
2. Answer the question using ONLY information from the context
3. If the answer cannot be found in the context, explicitly say "I cannot find this in the provided documents"
4. Provide accurate, concise answers
5. Include citations in the format [Document: filename, Page: X] for each piece of information used"""

# User prompt template
USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Data directories
PDFS_DIR = "data/pdfs"
VECTORSTORE_DIR = "vectorstore"