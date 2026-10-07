"""
Hybrid retrieval module for DocuMind.
Combines vector similarity search with BM25 keyword search for improved retrieval.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Dict, Tuple
import re

from config import (
    CHROMA_PERSIST_DIR, COLLECTION_NAME,
    EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE,
    TOP_K, TOP_K_BM25,
    VECTOR_WEIGHT, BM25_WEIGHT
)


class VectorRetriever:
    """Retrieves documents using vector similarity search."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    
    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict]:
        """
        Retrieve top-k chunks using vector similarity.
        
        Args:
            query: User query string
            k: Number of results to retrieve
            
        Returns:
            List of dictionaries with chunk text, metadata, and score
        """
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        retrieved = []
        for i in range(len(results['ids'][0])):
            retrieved.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'score': 1 - results['distances'][0][i],  # Convert distance to similarity
                'retrieval_method': 'vector'
            })
        
        return retrieved


class BM25Retriever:
    """Retrieves documents using BM25 keyword search."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        self.bm25_index = None
        self.documents = []
        self.metadatas = []
        self._build_index()
    
    def _build_index(self):
        """Build BM25 index from all documents in the collection."""
        print("Building BM25 index...")
        
        # Get all documents from collection
        count = self.collection.count()
        results = self.collection.get(
            include=['documents', 'metadatas']
        )
        
        self.documents = results['documents']
        self.metadatas = results['metadatas']
        
        # Tokenize documents for BM25
        tokenized_docs = [self._tokenize(doc) for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
        
        print(f"BM25 index built with {len(self.documents)} documents.")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def retrieve(self, query: str, k: int = TOP_K_BM25) -> List[Dict]:
        """
        Retrieve top-k chunks using BM25.
        
        Args:
            query: User query string
            k: Number of results to retrieve
            
        Returns:
            List of dictionaries with chunk text, metadata, and score
        """
        if self.bm25_index is None:
            self._build_index()
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        # Format results
        retrieved = []
        for idx in top_k_indices:
            retrieved.append({
                'text': self.documents[idx],
                'metadata': self.metadatas[idx],
                'score': float(scores[idx]),
                'retrieval_method': 'bm25'
            })
        
        return retrieved


class HybridRetriever:
    """Combines vector and BM25 retrieval with score fusion."""
    
    def __init__(self):
        self.vector_retriever = VectorRetriever()
        self.bm25_retriever = BM25Retriever()
    
    def retrieve(self, query: str, k: int = TOP_K, use_hybrid: bool = True) -> List[Dict]:
        """
        Retrieve chunks using hybrid or vector-only search.
        
        Args:
            query: User query string
            k: Number of results to return
            use_hybrid: If True, use hybrid (vector + BM25), else vector-only
            
        Returns:
            List of dictionaries with chunk text, metadata, and combined score
        """
        if not use_hybrid:
            # Vector-only retrieval
            results = self.vector_retriever.retrieve(query, k=k)
            return self._normalize_scores(results)
        
        # Hybrid retrieval
        vector_results = self.vector_retriever.retrieve(query, k=TOP_K_BM25)
        bm25_results = self.bm25_retriever.retrieve(query, k=TOP_K_BM25)
        
        # Merge and re-rank
        combined = self._merge_and_rerank(vector_results, bm25_results, k)
        
        return combined
    
    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Results with normalized scores
        """
        if not results:
            return results
        
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same
            for r in results:
                r['score'] = 1.0
        else:
            for r in results:
                r['score'] = (r['score'] - min_score) / (max_score - min_score)
        
        return results
    
    def _merge_and_rerank(self, vector_results: List[Dict], 
                         bm25_results: List[Dict], k: int) -> List[Dict]:
        """
        Merge vector and BM25 results using score fusion.
        
        Args:
            vector_results: Results from vector retrieval
            bm25_results: Results from BM25 retrieval
            k: Number of top results to return
            
        Returns:
            Merged and re-ranked results
        """
        # Normalize scores
        vector_results = self._normalize_scores(vector_results)
        bm25_results = self._normalize_scores(bm25_results)
        
        # Create a dictionary to track unique chunks by their ID
        merged = {}
        
        # Add vector results
        for result in vector_results:
            chunk_id = f"{result['metadata']['source']}_chunk_{result['metadata']['chunk_index']}"
            if chunk_id not in merged:
                merged[chunk_id] = {
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'vector_score': result['score'],
                    'bm25_score': 0.0,
                    'combined_score': result['score'] * VECTOR_WEIGHT
                }
            else:
                merged[chunk_id]['vector_score'] = result['score']
        
        # Add BM25 results
        for result in bm25_results:
            chunk_id = f"{result['metadata']['source']}_chunk_{result['metadata']['chunk_index']}"
            if chunk_id not in merged:
                merged[chunk_id] = {
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'vector_score': 0.0,
                    'bm25_score': result['score'],
                    'combined_score': result['score'] * BM25_WEIGHT
                }
            else:
                merged[chunk_id]['bm25_score'] = result['score']
                # Update combined score
                merged[chunk_id]['combined_score'] = (
                    merged[chunk_id]['vector_score'] * VECTOR_WEIGHT +
                    result['score'] * BM25_WEIGHT
                )
        
        # Sort by combined score
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        # Return top-k
        top_k = sorted_results[:k]
        
        # Format output
        formatted = []
        for result in top_k:
            formatted.append({
                'text': result['text'],
                'metadata': result['metadata'],
                'score': result['combined_score'],
                'vector_score': result['vector_score'],
                'bm25_score': result['bm25_score'],
                'retrieval_method': 'hybrid'
            })
        
        return formatted
    
    def get_context_string(self, results: List[Dict]) -> str:
        """
        Convert retrieval results to a context string for LLM prompting.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Formatted context string with citations
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            source = result['metadata']['source']
            page = result['metadata']['page']
            text = result['text']
            
            context_part = f"[{i}] Source: {source}, Page: {page}\n{text}"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)


# Main execution for testing
if __name__ == "__main__":
    retriever = HybridRetriever()
    
    # Test query
    query = "What is the main topic of the documents?"
    
    # Test vector-only retrieval
    print("\n=== Vector-only Retrieval ===")
    vector_results = retriever.retrieve(query, k=3, use_hybrid=False)
    for i, result in enumerate(vector_results, 1):
        print(f"\nResult {i}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['metadata']['source']}, Page: {result['metadata']['page']}")
        print(f"Text: {result['text'][:200]}...")
    
    # Test hybrid retrieval
    print("\n=== Hybrid Retrieval ===")
    hybrid_results = retriever.retrieve(query, k=3, use_hybrid=True)
    for i, result in enumerate(hybrid_results, 1):
        print(f"\nResult {i}:")
        print(f"Combined Score: {result['score']:.4f}")
        print(f"Vector Score: {result['vector_score']:.4f}")
        print(f"BM25 Score: {result['bm25_score']:.4f}")
        print(f"Source: {result['metadata']['source']}, Page: {result['metadata']['page']}")
        print(f"Text: {result['text'][:200]}...")
