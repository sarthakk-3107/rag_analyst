# 📊 Financial RAG Analyst

AI-powered Q&A for SEC 10-K filings using Retrieval-Augmented Generation.

## Features

- 📥 Download 10-K filings directly from SEC EDGAR
- 🔍 Hybrid search (semantic + keyword)
- 🤖 AI-generated answers with citations
- 📊 Automatic financial metric extraction
- 💾 Local caching of downloaded filings

## Tech Stack

- **LLM:** OpenAI GPT-5-mini
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector DB:** ChromaDB
- **Keyword Search:** BM25
- **UI:** Streamlit
- **Data Source:** SEC EDGAR

## Setup

### 1. Install Dependencies

```bash
cd rag_analyst
uv venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

uv pip install -r requirements.txt
```

### 2. Configure API Key

**Important:** Your OpenAI API key must have access to:
- `gpt-5-mini` (for LLM)
- `text-embedding-3-small` (for embeddings)

Create a `.env` file:

```
OPENAI_API_KEY=your-api-key-here
```

### 3. Run

```bash
streamlit run financial_rag.py
```

## Dependencies

- `openai` - LLM and embeddings
- `chromadb` - Vector database
- `rank-bm25` - Keyword search
- `streamlit` - Web UI
- `sec-edgar-downloader` - SEC filing downloads
- `beautifulsoup4` - HTML parsing
- `python-dotenv` - Environment variables
