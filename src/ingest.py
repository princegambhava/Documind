"""
PDF ingestion pipeline for DocuMind.
Handles PDF extraction, chunking, embedding generation, and vector storage.
"""

import os
import fitz  # PyMuPDF
from typing import List, Dict, Tuple
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP,
    SENTENCE_CHUNK_MAX_SIZE, SENTENCE_CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE,
    CHROMA_PERSIST_DIR, COLLECTION_NAME,
    USE_OCR_FALLBACK, OCR_DPI
)


class PDFExtractor:
    """Extract text from PDF files with OCR fallback for scanned documents."""
    
    def __init__(self):
        self.use_ocr = USE_OCR_FALLBACK
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, str]]:
        """
        Extract text from a PDF file, page by page.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries with 'page_num' and 'text' keys
        """
        try:
            doc = fitz.open(pdf_path)
            pages_text = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                # If page is empty or has very little text, try OCR
                if self.use_ocr and len(text.strip()) < 50:
                    text = self._extract_with_ocr(page, pdf_path, page_num)
                
                # Clean the extracted text
                text = self._clean_text(text)
                
                if text.strip():  # Only add non-empty pages
                    pages_text.append({
                        'page_num': page_num + 1,  # 1-indexed for user display
                        'text': text
                    })
            
            doc.close()
            return pages_text
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF {pdf_path}: {str(e)}")
    
    def _extract_with_ocr(self, page, pdf_path: str, page_num: int) -> str:
        """
        Fallback OCR extraction for scanned PDFs using pdf2image and pytesseract.
        
        Args:
            page: PyMuPDF page object
            pdf_path: Path to the PDF file
            page_num: Page number
            
        Returns:
            Extracted text string
        """
        try:
            from pdf2image import convert_from_path
            import pytesseract
            
            # Convert specific page to image
            images = convert_from_path(
                pdf_path, 
                first_page=page_num + 1, 
                last_page=page_num + 1,
                dpi=OCR_DPI
            )
            
            if images:
                text = pytesseract.image_to_string(images[0])
                return text
            
            return ""
            
        except ImportError:
            print("Warning: pdf2image or pytesseract not installed. OCR fallback disabled.")
            return ""
        except Exception as e:
            print(f"Warning: OCR extraction failed for page {page_num}: {str(e)}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing excessive whitespace and normalizing.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text


class Chunker:
    """Chunk text into smaller segments for embedding and retrieval."""
    
    def __init__(self):
        pass
    
    def chunk_fixed_size(self, pages_text: List[Dict[str, str]], filename: str) -> List[Dict]:
        """
        Chunk text using fixed-size windows with overlap.
        
        Args:
            pages_text: List of page dictionaries with 'page_num' and 'text'
            filename: Source filename for metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        chunk_id = 0
        
        for page in pages_text:
            text = page['text']
            page_num = page['page_num']
            
            # Split into fixed-size chunks with overlap
            start = 0
            while start < len(text):
                end = start + CHUNK_SIZE
                chunk_text = text[start:end]
                
                # Only add non-empty chunks
                if chunk_text.strip():
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': chunk_text,
                        'metadata': {
                            'source': filename,
                            'page': page_num,
                            'chunk_index': chunk_id,
                            'chunking_strategy': 'fixed_size'
                        }
                    })
                    chunk_id += 1
                
                start = end - CHUNK_OVERLAP
        
        return chunks
    
    def chunk_sentence_based(self, pages_text: List[Dict[str, str]], filename: str) -> List[Dict]:
        """
        Chunk text based on sentence boundaries.
        
        Args:
            pages_text: List of page dictionaries with 'page_num' and 'text'
            filename: Source filename for metadata
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        chunk_id = 0
        
        for page in pages_text:
            text = page['text']
            page_num = page['page_num']
            
            # Split into sentences using regex
            sentences = re.split(r'(?<=[.!?])\s+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Group sentences into chunks
            current_chunk = ""
            current_sentences = []
            
            for i, sentence in enumerate(sentences):
                # Check if adding this sentence would exceed max size
                if len(current_chunk) + len(sentence) > SENTENCE_CHUNK_MAX_SIZE and current_chunk:
                    # Save current chunk
                    chunks.append({
                        'chunk_id': chunk_id,
                        'text': current_chunk.strip(),
                        'metadata': {
                            'source': filename,
                            'page': page_num,
                            'chunk_index': chunk_id,
                            'chunking_strategy': 'sentence_based'
                        }
                    })
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    current_sentences = current_sentences[-SENTENCE_CHUNK_OVERLAP:] if SENTENCE_CHUNK_OVERLAP > 0 else []
                    current_chunk = " ".join(current_sentences) + " " if current_sentences else ""
                
                current_chunk += sentence + " "
                current_sentences.append(sentence)
            
            # Add the last chunk if it has content
            if current_chunk.strip():
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': current_chunk.strip(),
                    'metadata': {
                        'source': filename,
                        'page': page_num,
                        'chunk_index': chunk_id,
                        'chunking_strategy': 'sentence_based'
                    }
                })
                chunk_id += 1
        
        return chunks


class Embedder:
    """Generate embeddings for text chunks using sentence-transformers."""
    
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
        print("Embedding model loaded successfully.")
    
    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            numpy array of embeddings
        """
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings


class VectorStore:
    """Manage ChromaDB vector store for document chunks."""
    
    def __init__(self):
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray):
        """
        Add chunks and their embeddings to the vector store.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: numpy array of embeddings
        """
        ids = [f"{chunk['metadata']['source']}_chunk_{chunk['chunk_id']}" for chunk in chunks]
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Convert embeddings to list for ChromaDB
        embeddings_list = embeddings.tolist()
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"Added {len(chunks)} chunks to vector store.")
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print("Vector store cleared.")
        except Exception as e:
            print(f"Error clearing vector store: {str(e)}")
    
    def get_collection_info(self) -> Dict:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection stats
        """
        return {
            'count': self.collection.count(),
            'name': COLLECTION_NAME
        }


class IngestionPipeline:
    """Main pipeline for PDF ingestion: extract -> chunk -> embed -> store."""
    
    def __init__(self):
        self.extractor = PDFExtractor()
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore()
    
    def ingest_pdf(self, pdf_path: str, chunking_strategy: str = "fixed_size") -> int:
        """
        Ingest a single PDF file into the vector store.
        
        Args:
            pdf_path: Path to the PDF file
            chunking_strategy: Either "fixed_size" or "sentence_based"
            
        Returns:
            Number of chunks added
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        filename = os.path.basename(pdf_path)
        print(f"\nProcessing PDF: {filename}")
        
        # Step 1: Extract text from PDF
        print("Extracting text from PDF...")
        pages_text = self.extractor.extract_text_from_pdf(pdf_path)
        
        if not pages_text:
            raise ValueError(f"No text extracted from PDF: {pdf_path}")
        
        total_pages = len(pages_text)
        print(f"Extracted text from {total_pages} pages.")
        
        # Step 2: Chunk the text
        print(f"Chunking text using {chunking_strategy} strategy...")
        if chunking_strategy == "fixed_size":
            chunks = self.chunker.chunk_fixed_size(pages_text, filename)
        elif chunking_strategy == "sentence_based":
            chunks = self.chunker.chunk_sentence_based(pages_text, filename)
        else:
            raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")
        
        print(f"Created {len(chunks)} chunks.")
        
        # Step 3: Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedder.embed_chunks(chunks)
        
        # Step 4: Store in vector database
        print("Storing chunks in vector store...")
        self.vector_store.add_chunks(chunks, embeddings)
        
        print(f"Successfully ingested {filename}: {len(chunks)} chunks added.")
        return len(chunks)
    
    def ingest_directory(self, pdf_dir: str, chunking_strategy: str = "fixed_size") -> int:
        """
        Ingest all PDF files from a directory.
        
        Args:
            pdf_dir: Path to directory containing PDFs
            chunking_strategy: Either "fixed_size" or "sentence_based"
            
        Returns:
            Total number of chunks added
        """
        if not os.path.exists(pdf_dir):
            raise FileNotFoundError(f"Directory not found: {pdf_dir}")
        
        pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            return 0
        
        print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
        
        total_chunks = 0
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            try:
                chunks_added = self.ingest_pdf(pdf_path, chunking_strategy)
                total_chunks += chunks_added
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")
                continue
        
        print(f"\nIngestion complete. Total chunks added: {total_chunks}")
        return total_chunks
    
    def clear_vector_store(self):
        """Clear all data from the vector store."""
        self.vector_store.clear_collection()


# Main execution for testing
if __name__ == "__main__":
    pipeline = IngestionPipeline()
    
    # Example usage
    pdf_directory = "../data/pdfs"
    
    # Clear existing data (optional)
    # pipeline.clear_vector_store()
    
    # Ingest PDFs with fixed-size chunking
    total_chunks = pipeline.ingest_directory(pdf_directory, chunking_strategy="fixed_size")
    
    # Get collection info
    info = pipeline.vector_store.get_collection_info()
    print(f"\nVector store now contains {info['count']} chunks.")
