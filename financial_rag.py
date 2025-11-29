"""
Financial Filings RAG Analyst Assistant - Main Application
FE524 Project - Phase 1

Dependencies (using uv):
uv venv
.venv\\Scripts\\activate  # Windows
uv pip install -r requirements.txt

Or create a virtual environment:
uv venv
.venv\\Scripts\\activate  (Windows) or source .venv/bin/activate (Mac/Linux)
uv pip install -r requirements.txt

API key setup:
Create a .env file in this directory with:
OPENAI_API_KEY=your-key-here

Then run: streamlit run financial_rag.py
"""

import os
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import numpy as np
from openai import OpenAI
import streamlit as st
from sec_edgar_downloader import Downloader

# Import document processor
try:
    from document_processor import SECDocumentProcessor
except ImportError:
    st.error("⚠️ document_processor.py not found! Please ensure it's in the same directory.")
    st.stop()

@dataclass
class DocumentChunk:
    """Represents a chunk of a financial document"""
    text: str
    source: str
    metadata: Dict[str, Any]
    embedding: np.ndarray = None

class FinancialRAGSystem:
    """Main RAG system for financial document analysis"""

    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        # Using OpenAI's text-embedding-3-small for better semantic search
        self.embedding_model_name = "text-embedding-3-small"
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = None
        self.bm25 = None
        self.chunks = []
        self.doc_processor = SECDocumentProcessor()

    def _truncate_text(self, text: str, max_chars: int = 8000) -> str:
        """Truncate text to fit within embedding model token limits.
        
        text-embedding-3-small has 8192 token limit.
        Using conservative estimate: ~1 token per 3 chars.
        8000 chars ≈ 2600 tokens (safe margin under 8192 limit).
        """
        if len(text) > max_chars:
            return text[:max_chars]
        return text

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI API (text-embedding-3-small)"""
        # Truncate texts to fit within model's token limit
        truncated_texts = [self._truncate_text(t) for t in texts]
        
        # Small batch size to avoid hitting API limits
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model_name,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                # If batch fails, try one at a time with more aggressive truncation
                for text in batch:
                    try:
                        # Even more aggressive truncation for problematic texts
                        safe_text = text[:5000] if len(text) > 5000 else text
                        response = self.client.embeddings.create(
                            model=self.embedding_model_name,
                            input=[safe_text]
                        )
                        all_embeddings.append(response.data[0].embedding)
                    except Exception as inner_e:
                        # Last resort: very short text
                        response = self.client.embeddings.create(
                            model=self.embedding_model_name,
                            input=[text[:2000]]
                        )
                        all_embeddings.append(response.data[0].embedding)
        
        return all_embeddings

    def download_10k(self, ticker: str, email: str, num_filings: int = 1):
        """Download SEC 10-K filings for a company.
        
        Returns:
            tuple: (filepath, was_cached) - filepath to the filing, and whether it was already cached
        """
        filing_dir = f"sec-edgar-filings/{ticker}/10-K"
        
        # Check if filing already exists locally
        if os.path.exists(filing_dir):
            files = list(Path(filing_dir).rglob("*.txt"))
            if files:
                # Filing already downloaded, return existing file
                return str(files[0]), True  # True = already cached
        
        # Download if not found locally
        dl = Downloader("FE524-Project", email)
        dl.get("10-K", ticker, limit=num_filings)
        
        # Find the downloaded filing
        if os.path.exists(filing_dir):
            files = list(Path(filing_dir).rglob("*.txt"))
            if files:
                return str(files[0]), False  # False = newly downloaded
        return None, False

    def process_and_index_filing(self, filepath: str) -> int:
        """Process SEC filing and create searchable index"""
        # Process the filing
        sections, metadata = self.doc_processor.process_filing(filepath)

        # Create chunks from all sections
        all_chunks = []
        
        # Check parsing method and provide feedback
        parsing_method = metadata.get('parsing_method', 'unknown')
        
        if parsing_method == 'structured':
            st.success(f"✅ Successfully parsed {len(sections)} document sections")
        elif parsing_method == 'fallback_keyword':
            st.info(f"ℹ️ Used keyword-based parsing. Found {len(sections)} content sections.")
        elif parsing_method == 'full_document':
            st.warning("⚠️ Section headers not found. Using full document chunking.")

        if sections and len(sections) > 0:
            # Process each section
            for section in sections:
                section_chunks = self.doc_processor.chunk_section(section)

                # Convert to DocumentChunk objects
                for chunk_dict in section_chunks:
                    chunk = DocumentChunk(
                        text=chunk_dict['text'],
                        source=f"{chunk_dict['metadata']['section']} - {chunk_dict['metadata']['section_title']}",
                        metadata=chunk_dict['metadata']
                    )
                    all_chunks.append(chunk)
        else:
            # Final fallback: chunk the entire document
            raw_content = self.doc_processor.read_filing(filepath)
            clean_text = self.doc_processor.clean_html(raw_content)

            # Create chunks from full text (smaller chunks to fit embedding model limits)
            words = clean_text.split()
            chunk_size = 500
            overlap = 100

            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = ' '.join(words[i:i + chunk_size])
                if len(chunk_text) > 100:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        source=f"{metadata.get('company_name', 'Unknown')} 10-K",
                        metadata={
                            'company': metadata.get('company_name', 'Unknown'),
                            'filing_date': metadata.get('filing_date', ''),
                            'chunk_id': len(all_chunks)
                        }
                    )
                    all_chunks.append(chunk)

        if len(all_chunks) == 0:
            raise ValueError("No document chunks created. The file may be empty or corrupted.")

        # Index the chunks
        num_chunks = self.index_documents(all_chunks)
        return num_chunks, metadata

    def index_documents(self, chunks: List[DocumentChunk]):
        """Create vector and BM25 indexes for document chunks"""
        self.chunks = chunks

        # Create vector embeddings using OpenAI API
        texts = [chunk.text for chunk in chunks]
        embeddings = self.get_embeddings(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = np.array(embedding)

        # Initialize ChromaDB collection
        try:
            self.chroma_client.delete_collection("financial_docs")
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name="financial_docs",
            metadata={"hnsw:space": "cosine"}
        )

        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,  # Already a list from OpenAI API
            documents=texts,
            metadatas=[chunk.metadata for chunk in chunks],
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )

        # Create BM25 index
        tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        return len(chunks)

    def hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid retrieval using dense embeddings and BM25"""

        # Dense retrieval (semantic search) using OpenAI embeddings
        query_embedding = self.get_embeddings([query])[0]
        dense_results = self.collection.query(
            query_embeddings=[query_embedding],  # Already a list from OpenAI API
            n_results=top_k * 2
        )

        # BM25 retrieval (keyword search)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[-top_k * 2:][::-1]

        # Combine and rerank
        chunk_scores = {}

        # Add dense results
        for idx, doc_id in enumerate(dense_results['ids'][0]):
            chunk_idx = int(doc_id.split('_')[1])
            chunk_scores[chunk_idx] = chunk_scores.get(chunk_idx, 0) + (1 - idx / (top_k * 2))

        # Add BM25 results
        for rank, idx in enumerate(bm25_top_indices):
            chunk_scores[idx] = chunk_scores.get(idx, 0) + (1 - rank / (top_k * 2))

        # Get top-k by combined score
        top_indices = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        retrieved_chunks = []
        for idx, score in top_indices:
            chunk = self.chunks[idx]
            retrieved_chunks.append({
                'text': chunk.text,
                'source': chunk.source,
                'metadata': chunk.metadata,
                'relevance_score': score
            })

        return retrieved_chunks

    def calculate_financial_metrics(self, context: str, query: str) -> Dict[str, Any]:
        """Use LLM to extract and calculate financial metrics"""

        calculation_prompt = f"""You are a financial analyst. Based on the following context, 
extract relevant financial data and perform calculations to answer the query.

Context:
{context}

Query: {query}

Provide:
1. Extracted financial numbers with their sources
2. Step-by-step calculations
3. Final computed metrics as a JSON object

Format your response as JSON:
{{
    "extracted_data": {{"item": value}},
    "calculations": ["step 1", "step 2"],
    "metrics": {{"metric_name": value}},
    "insights": ["insight 1", "insight 2"]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",  # Upgraded from gpt-4o-mini for better financial analysis
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert at extracting data and performing calculations."},
                    {"role": "user", "content": calculation_prompt}
                ],
                temperature=0.2
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {
                "error": str(e),
                "calculations": [],
                "metrics": {},
                "insights": []
            }

    def generate_answer(self, query: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate final answer with citations"""

        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"[Source {i+1}: {chunk['source']}]\n{chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        ])

        # Calculate metrics if needed
        metrics_result = self.calculate_financial_metrics(context, query)

        # Generate narrative answer
        answer_prompt = f"""You are a financial analyst assistant. Answer the user's question based on the provided context.

Context from SEC 10-K filings:
{context}

Calculated Metrics:
{json.dumps(metrics_result.get('metrics', {}), indent=2)}

Question: {query}

Provide a clear, professional answer that:
1. Directly answers the question
2. Cites specific sources (e.g., "According to [Source 1]...")
3. Includes relevant metrics and calculations
4. Highlights key insights

Keep your answer concise but comprehensive."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",  # Upgraded from gpt-4o-mini for better financial analysis
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst providing evidence-based insights."},
                    {"role": "user", "content": answer_prompt}
                ],
                temperature=0.3
            )

            return {
                'answer': response.choices[0].message.content,
                'retrieved_chunks': retrieved_chunks,
                'metrics': metrics_result.get('metrics', {}),
                'calculations': metrics_result.get('calculations', []),
                'insights': metrics_result.get('insights', [])
            }
        except Exception as e:
            return {
                'error': f"Error generating answer: {str(e)}",
                'retrieved_chunks': retrieved_chunks,
                'metrics': {},
                'calculations': [],
                'insights': []
            }

    def query(self, question: str) -> Dict[str, Any]:
        """Main query interface"""
        if not self.chunks:
            return {"error": "No documents indexed. Please load documents first."}

        # Retrieve relevant chunks
        retrieved = self.hybrid_retrieve(question, top_k=5)

        # Generate answer
        result = self.generate_answer(question, retrieved)

        return result


# Streamlit UI
def main():
    st.set_page_config(page_title="Financial RAG Analyst", page_icon="📊", layout="wide")

    st.title("📊 Financial RAG Analyst")
    st.markdown("*Ask questions about company 10-K filings and get AI-powered insights*")

    # Check for API key in environment
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        st.error("❌ OpenAI API key not found!")
        st.info("""
        Create a `.env` file in the project folder with:
        ```
        OPENAI_API_KEY=your-key-here
        ```
        Then restart the app.
        """)
        st.stop()

    # Sidebar for configuration
    with st.sidebar:
        st.header("📁 Load Company Filing")
        ticker = st.text_input("Company Ticker", "AAPL", help="e.g., AAPL, MSFT, TSLA, GOOGL")
        email = st.text_input("Your Email", "user@email.com", help="Required by SEC for downloading")

        if st.button("📥 Load 10-K Filing", type="primary", use_container_width=True):
            with st.spinner(f"Loading {ticker} 10-K..."):
                try:
                    # Initialize or get RAG system
                    if 'rag_system' not in st.session_state:
                        st.session_state['rag_system'] = FinancialRAGSystem(api_key)

                    rag_system = st.session_state['rag_system']

                    # Check for existing or download filing
                    filepath, was_cached = rag_system.download_10k(ticker, email)

                    if not filepath:
                        st.error("Could not find filing. Check the ticker symbol.")
                    else:
                        # Show whether using cached or newly downloaded
                        if was_cached:
                            st.info(f"📂 Using cached {ticker} filing")
                        else:
                            st.success(f"📥 Downloaded {ticker} filing")
                        
                        # Process and index
                        num_chunks, metadata = rag_system.process_and_index_filing(filepath)

                        # Store metadata in session
                        st.session_state['metadata'] = metadata
                        st.session_state['ticker'] = ticker

                        st.success(f"✅ Ready! {num_chunks} sections indexed")
                        st.balloons()

                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        # Show loaded document info
        if 'metadata' in st.session_state:
            st.markdown("---")
            st.subheader("📄 Loaded Document")
            metadata = st.session_state['metadata']
            st.write(f"**{metadata.get('company_name', st.session_state.get('ticker', 'Unknown'))}**")
            st.caption(f"Filed: {metadata.get('filing_date', 'N/A')}")

        st.markdown("---")
        st.header("📋 Sample Questions")
        sample_questions = [
            "What was the total revenue?",
            "Calculate the operating margin",
            "What are the main risk factors?",
            "Compare R&D expenses to revenue",
            "What are the key business segments?"
        ]

        for q in sample_questions:
            if st.button(q, key=f"sample_{q}"):
                st.session_state['query'] = q
                st.session_state['auto_analyze'] = True  # Flag to auto-run analysis
                st.rerun()  # Rerun to trigger analysis

    # Show welcome message if no document loaded
    if 'rag_system' not in st.session_state:
        st.info("👈 **Get started:** Enter a company ticker in the sidebar and click 'Load 10-K Filing'")
        st.markdown("---")
    
    # Main query interface
    st.header("💬 Ask a Question")

    # Get query from session state or text area
    default_query = st.session_state.get('query', '')

    query = st.text_area(
        "What would you like to know?",
        value=default_query,
        height=80,
        placeholder="e.g., What was the total revenue? What are the main risks?",
        key="query_input"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # Check if auto-analyze was triggered by sample question
    auto_analyze = st.session_state.pop('auto_analyze', False)
    
    # Process query (either from button click or auto-analyze)
    if (analyze_btn or auto_analyze) and query:
        if 'rag_system' not in st.session_state:
            st.warning("Please load a 10-K filing first using the sidebar.")
        else:
            rag_system = st.session_state['rag_system']

            with st.spinner("Analyzing..."):
                result = rag_system.query(query)

                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    # Display answer
                    st.header("📝 Answer")
                    st.markdown(result['answer'])

                    # Display metrics
                    if result.get('metrics') and len(result['metrics']) > 0:
                        st.header("📊 Computed Metrics")
                        metric_cols = st.columns(min(len(result['metrics']), 4))
                        for idx, (metric, value) in enumerate(result['metrics'].items()):
                            with metric_cols[idx % 4]:
                                st.metric(label=metric, value=value)

                    # Display calculations
                    if result.get('calculations') and len(result['calculations']) > 0:
                        st.header("🧮 Calculations")
                        for i, calc in enumerate(result['calculations'], 1):
                            st.markdown(f"{i}. {calc}")

                    # Display insights
                    if result.get('insights') and len(result['insights']) > 0:
                        st.header("💡 Key Insights")
                        for insight in result['insights']:
                            st.markdown(f"- {insight}")

                    # Display retrieved chunks
                    st.header("📄 Sources")

                    for i, chunk in enumerate(result['retrieved_chunks'], 1):
                        with st.expander(
                            f"📑 Source {i}: {chunk['source']} | Relevance: {chunk['relevance_score']:.2%}",
                            expanded=(i == 1)
                        ):
                            st.markdown(chunk['text'])
                            st.caption(f"Company: {chunk['metadata'].get('company', 'N/A')} | Section: {chunk['metadata'].get('section', 'N/A')}")

    # Footer
    st.markdown("---")
    st.caption("Financial RAG Analyst • Powered by OpenAI • Data from SEC EDGAR")

if __name__ == "__main__":
    main()
