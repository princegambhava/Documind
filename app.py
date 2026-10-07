"""
Streamlit frontend for DocuMind RAG chatbot.
Provides a web interface for PDF upload, chat, and source citations.
"""

import streamlit as st
import os
import sys
from typing import List, Dict

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import src
from src import IngestionPipeline, HybridRetriever, AnswerGenerator, PDFS_DIR, LLM_BACKEND


# Page configuration
st.set_page_config(
    page_title="DocuMind - PDF Q&A Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .citation {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'ingestion_pipeline' not in st.session_state:
        st.session_state.ingestion_pipeline = None
    
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    
    if 'vector_store_loaded' not in st.session_state:
        st.session_state.vector_store_loaded = False
    
    if 'chunking_strategy' not in st.session_state:
        st.session_state.chunking_strategy = "fixed_size"
    
    if 'retrieval_strategy' not in st.session_state:
        st.session_state.retrieval_strategy = "hybrid"


def load_vector_store():
    """Load the vector store and initialize retriever/generator."""
    try:
        st.session_state.retriever = HybridRetriever()
        st.session_state.generator = AnswerGenerator()
        st.session_state.vector_store_loaded = True
        
        # Get collection info
        num_chunks = st.session_state.retriever.vector_retriever.collection.count()
        st.session_state.num_chunks = num_chunks
        
        return True, f"Vector store loaded with {num_chunks} chunks."
    except Exception as e:
        return False, f"Error loading vector store: {str(e)}"


def handle_pdf_upload(uploaded_files):
    """Handle PDF file uploads and ingestion."""
    if not uploaded_files:
        return False, "No files uploaded."
    
    # Save uploaded files to data/pdfs directory
    os.makedirs(PDFS_DIR, exist_ok=True)
    
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(PDFS_DIR, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    
    # Initialize ingestion pipeline
    st.session_state.ingestion_pipeline = IngestionPipeline()
    
    # Clear existing vector store (optional - can be made configurable)
    st.session_state.ingestion_pipeline.clear_vector_store()
    
    # Ingest PDFs
    total_chunks = 0
    chunking_strategy = st.session_state.chunking_strategy
    
    for pdf_path in saved_paths:
        try:
            chunks_added = st.session_state.ingestion_pipeline.ingest_pdf(
                pdf_path, 
                chunking_strategy=chunking_strategy
            )
            total_chunks += chunks_added
        except Exception as e:
            return False, f"Error processing {os.path.basename(pdf_path)}: {str(e)}"
    
    # Reload vector store
    success, message = load_vector_store()
    
    if success:
        return True, f"Successfully ingested {len(uploaded_files)} PDF(s) with {total_chunks} chunks."
    else:
        return False, message


def display_chat_message(role: str, content: str, citations: List[Dict] = None):
    """
    Display a chat message with optional citations.
    
    Args:
        role: Either 'user' or 'assistant'
        content: Message content
        citations: List of citation dictionaries
    """
    with st.chat_message(role):
        st.markdown(content)
        
        # Display citations if available
        if citations and role == 'assistant':
            st.markdown("---")
            st.markdown("**📚 Sources:**")
            for i, citation in enumerate(citations, 1):
                st.markdown(f"{i}. 📄 **{citation['source']}** - Page {citation['page']}")


def main():
    """Main Streamlit app function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📚 DocuMind</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">PDF Question-Answering Chatbot with RAG</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Chunking strategy selection
        st.subheader("Chunking Strategy")
        chunking_option = st.radio(
            "Select chunking method:",
            ["fixed_size", "sentence_based"],
            index=0,
            help="Fixed-size: Equal length chunks with overlap. Sentence-based: Chunks based on sentence boundaries."
        )
        st.session_state.chunking_strategy = chunking_option
        
        # Retrieval strategy selection
        st.subheader("Retrieval Strategy")
        retrieval_option = st.radio(
            "Select retrieval method:",
            ["hybrid", "vector"],
            index=0,
            help="Hybrid: Combines vector similarity + BM25 keyword search. Vector: Vector similarity only."
        )
        st.session_state.retrieval_strategy = retrieval_option
        
        # LLM backend info
        st.subheader("LLM Backend")
        st.info(f"Current backend: {LLM_BACKEND}")
        st.caption("Configure in .env file (LLM_BACKEND=gemini or groq)")
        
        # Vector store status
        st.subheader("Vector Store Status")
        if st.session_state.vector_store_loaded:
            st.success(f"✅ Loaded ({st.session_state.num_chunks} chunks)")
        else:
            st.warning("⚠️ Not loaded")
        
        # PDF Upload section
        st.subheader("📤 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to ingest into the system."
        )
        
        if st.button("Ingest PDFs", type="primary"):
            if uploaded_files:
                with st.spinner("Processing PDFs..."):
                    success, message = handle_pdf_upload(uploaded_files)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            else:
                st.warning("Please upload at least one PDF file.")
        
        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # Clear vector store button
        if st.button("Clear Vector Store"):
            if st.session_state.ingestion_pipeline:
                st.session_state.ingestion_pipeline.clear_vector_store()
                st.session_state.vector_store_loaded = False
                st.session_state.messages = []
                st.success("Vector store cleared.")
            else:
                st.warning("No ingestion pipeline initialized.")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat interface
        st.header("💬 Chat")
        
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(
                message['role'],
                message['content'],
                message.get('citations')
            )
        
        # Add some spacing before chat input
        st.markdown("<br>", unsafe_allow_html=True)
    
    with col2:
        # Info panel
        st.header("ℹ️ Info")
        
        st.markdown("""
        **How to use:**
        1. Upload PDFs in the sidebar
        2. Click "Ingest PDFs"
        3. Ask questions in the chat
        4. View answers with citations
        
        **Features:**
        - Multi-PDF support
        - Hybrid retrieval (vector + BM25)
        - Source citations
        - Configurable chunking
        - Two LLM backends
        """)
        
        st.markdown("---")
        
        st.markdown("""
        **Current Settings:**
        - Chunking: `{chunking_option}`
        - Retrieval: `{retrieval_option}`
        - Backend: `{LLM_BACKEND}`
        """.format(
            chunking_option=chunking_option,
            retrieval_option=retrieval_option,
            LLM_BACKEND=LLM_BACKEND
        ))
    
    # Chat input (must be outside columns)
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if vector store is loaded
        if not st.session_state.vector_store_loaded:
            st.error("Please upload and ingest PDFs first.")
            st.stop()
        
        # Add user message to chat history
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt
        })
        display_chat_message('user', prompt)
        
        # Process the query
        with st.spinner("Searching documents and generating answer..."):
            try:
                # Retrieve relevant chunks
                use_hybrid = (st.session_state.retrieval_strategy == "hybrid")
                retrieval_results = st.session_state.retriever.retrieve(
                    prompt,
                    k=5,
                    use_hybrid=use_hybrid
                )
                
                # Build context
                context = st.session_state.retriever.get_context_string(retrieval_results)
                
                # Generate answer
                result = st.session_state.generator.generate(
                    context,
                    prompt,
                    retrieval_results
                )
                
                # Add assistant message to chat history
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': result['answer'],
                    'citations': result['citations'],
                    'backend': result['backend']
                })
                
                # Display assistant response
                display_chat_message(
                    'assistant',
                    result['answer'],
                    result['citations']
                )
                
                # Display backend info
                st.caption(f"Generated using: {result['backend']}")
                
            except Exception as e:
                error_message = f"Error processing your question: {str(e)}"
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': error_message
                })
                st.error(error_message)


if __name__ == "__main__":
    main()
