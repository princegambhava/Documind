# DocuMind - PDF Question-Answering Chatbot

A full-stack RAG (Retrieval-Augmented Generation) based PDF Question-Answering chatbot built as a final year college project. Users can upload PDF documents and ask natural language questions about their content. The system retrieves relevant chunks using semantic search and generates accurate, source-grounded answers with citations.

## 🏗️ Architecture

DocuMind follows a modular RAG pipeline architecture:

```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Storage
                                                              ↓
User Query ← Answer Generation ← LLM (Gemini/Groq) ← Context Building ← Hybrid Retrieval
```

### Components

1. **Ingestion Pipeline** (`src/ingest.py`)
   - PDF text extraction using PyMuPDF (fitz)
   - OCR fallback for scanned PDFs (pdf2image + pytesseract)
   - Two chunking strategies: fixed-size and sentence-based
   - Embedding generation using sentence-transformers
   - Persistent vector storage with ChromaDB

2. **Retrieval System** (`src/retriever.py`)
   - Vector similarity search using sentence-transformers embeddings
   - BM25 keyword search using rank-bm25
   - Hybrid retrieval with score fusion (configurable weights)
   - Re-ranking of combined results

3. **Answer Generation** (`src/generator.py`)
   - LLM API integration (Gemini and Groq)
   - Automatic backend selection and fallback
   - Retry logic with exponential backoff for rate limits
   - Citation extraction from retrieved chunks

4. **Evaluation Module** (`src/evaluate.py`)
   - Automated testing on Q&A pairs
   - Recall@k metrics for retrieval
   - Correctness scoring for generation
   - Comparison of all strategy combinations

5. **Frontend** (`app.py`)
   - Streamlit-based web interface
   - Multi-PDF upload support
   - Chat history with citations
   - Configurable chunking and retrieval strategies

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Google account (for Gemini API key)
- Groq account (optional, for backup LLM backend)

### Step 1: Clone/Download the Project

```bash
cd DocuMind2
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Get Free API Keys

#### Gemini API Key (Primary Backend)

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in with your Google account
3. Navigate to API Keys section
4. Create a new API key (free tier, no credit card required)
5. Copy the API key

#### Groq API Key (Optional Backup Backend)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key (free tier available)
5. Copy the API key

### Step 5: Configure Environment Variables

The project uses a `.env` file for configuration. The `.env` file has already been created with your Gemini API key:

```
GOOGLE_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
LLM_BACKEND=gemini
```

**Note:** Your Gemini API key has already been added to the `.env` file. If you want to use Groq as a backup, add your Groq API key.

### Step 6: Install Tesseract for OCR (Optional)

If you want OCR support for scanned PDFs:

**Windows:**
1. Download Tesseract from [GitHub Releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install and add to system PATH

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

## 📖 Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

**Using the App:**
1. Upload one or more PDF files using the sidebar
2. Click "Ingest PDFs" to process the documents
3. Select your preferred chunking and retrieval strategies
4. Ask questions in the chat interface
5. View answers with source citations

### Running the Evaluation Script

```bash
python src/evaluate.py
```

The evaluation script will:
- Load test Q&A pairs from `data/eval_qa.json`
- Test all combinations of chunking and retrieval strategies
- Calculate recall@k and correctness metrics
- Generate a summary table and CSV report

**Customizing Test Data:**
Edit `data/eval_qa.json` to add your own test Q&A pairs. The format is:
```json
[
  {
    "question": "Your question here",
    "answer": "Expected answer",
    "source": "document_name.pdf"
  }
]
```

## ⚙️ Configuration

All configurable parameters are in `src/config.py`:

- **Chunking:** `CHUNK_SIZE`, `CHUNK_OVERLAP`, `SENTENCE_CHUNK_MAX_SIZE`
- **Retrieval:** `TOP_K`, `VECTOR_WEIGHT`, `BM25_WEIGHT`
- **Models:** `EMBEDDING_MODEL_NAME`, `GEMINI_MODEL`, `GROQ_MODEL`
- **LLM Backend:** `LLM_BACKEND` (set in `.env` file)
- **Evaluation:** `EVAL_TOP_K_VALUES`, `EVAL_DELAY_SECONDS`

## 🧪 Testing PDF Processing

To test the ingestion pipeline independently:

```bash
python src/ingest.py
```

This will:
- Process all PDFs in `data/pdfs/`
- Use fixed-size chunking by default
- Display progress and statistics

## 📁 Project Structure

```
DocuMind2/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration constants
│   ├── ingest.py            # PDF extraction + chunking + embedding + storage
│   ├── retriever.py         # Hybrid retrieval logic
│   ├── generator.py         # LLM calling (Gemini/Groq)
│   └── evaluate.py          # Evaluation harness
├── data/
│   ├── pdfs/                # Upload PDFs here
│   └── eval_qa.json         # Test Q&A pairs
├── vectorstore/             # ChromaDB persistent storage (auto-created)
├── app.py                   # Streamlit frontend
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API keys)
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## 🔧 Key Features

### Multi-PDF Support
- Upload and process multiple PDFs simultaneously
- Track metadata per chunk (source filename, page number, chunk index)

### Dual Chunking Strategies
- **Fixed-size:** Equal length chunks with configurable overlap
- **Sentence-based:** Chunks based on sentence boundaries for better semantic coherence

### Hybrid Retrieval
- **Vector search:** Semantic similarity using sentence-transformers embeddings
- **BM25:** Keyword-based search for exact matches
- **Score fusion:** Combines both methods with configurable weights

### Dual LLM Backends
- **Gemini (primary):** Google's gemini-1.5-flash model with generous free tier
- **Groq (backup):** openai/gpt-oss-20b with automatic fallback
- **Rate limit handling:** Automatic retry with exponential backoff

### Source Citations
- Every answer includes document name and page number
- Helps verify information and locate sources

### Evaluation Framework
- Automated testing on custom Q&A pairs
- Recall@k metrics for retrieval quality
- Correctness scoring for answer quality
- Comparison matrix for all strategy combinations

## 🐛 Troubleshooting

### Common Issues

**"GOOGLE_API_KEY not set"**
- Ensure your API key is in the `.env` file
- Restart the Streamlit app after changing `.env`

**"No text extracted from PDF"**
- The PDF might be scanned/image-based
- Install Tesseract for OCR support
- Check `USE_OCR_FALLBACK = True` in config.py

**"Rate limit hit" errors**
- Free tiers have per-minute/per-day limits
- The system automatically retries with backoff
- Switch to Groq backend if Gemini limits are exhausted
- Increase `EVAL_DELAY_SECONDS` for evaluation

**"Vector store not loaded"**
- Upload and ingest PDFs first
- Check that `vectorstore/` directory exists
- Clear and re-ingest if corrupted

**Import errors**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate your virtual environment
- Check Python version (3.8+ required)

## 📊 Evaluation Metrics

The evaluation script reports:

- **Recall@k:** Fraction of relevant documents retrieved in top-k results
- **Correctness:** Simple heuristic based on phrase overlap between generated and expected answers
- **Combined score:** Average of recall@5 and correctness for ranking configurations

## 🎓 Project Report / Viva Points

### Architecture Highlights
- Modular design with clear separation of concerns
- Configurable parameters for easy experimentation
- Persistent vector storage for efficiency
- Dual backend support for reliability

### Technical Decisions
- **ChromaDB:** Chosen for local persistence and ease of use
- **sentence-transformers:** Local embeddings, no API costs
- **Hybrid retrieval:** Combines semantic and keyword matching
- **Streamlit:** Rapid prototyping and clean UI

### Error Handling
- Corrupted PDF detection
- Empty document handling
- Missing API key validation
- Rate limit retry logic
- OCR fallback for scanned PDFs

### Future Enhancements
- More sophisticated chunking (semantic, recursive)
- Advanced reranking (cross-encoder)
- Multi-modal support (images, tables)
- User feedback integration
- Caching for repeated queries

## 📝 License

This is a final year college project. Feel free to use and modify for educational purposes.

## 🤝 Contributing

This is an academic project. For suggestions or improvements, please refer to your project supervisor.

## 📧 Support

For issues related to:
- **API keys:** Check the respective provider's documentation
- **Dependencies:** Ensure all packages are correctly installed
- **PDF processing:** Verify PDF files are not corrupted
- **Code errors:** Check error messages and consult inline comments

---

**Built with ❤️ for academic purposes**
