"""
Financial Filings RAG Analyst Assistant - Main Application
FE524 Project - Phase 1

Dependencies:
pip install openai chromadb sentence-transformers rank-bm25 sec-edgar-downloader streamlit pandas numpy beautifulsoup4 lxml

Set your API key before running:
Windows CMD: set OPENAI_API_KEY=your-key-here
Windows PowerShell: $env:OPENAI_API_KEY="your-key-here"
Mac/Linux: export OPENAI_API_KEY=your-key-here

Then run: streamlit run financial_rag_app.py
"""

import os
import json
from typing import List, Dict, Any
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np
from openai import OpenAI
import streamlit as st
from sec_edgar_downloader import Downloader
from pathlib import Path

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
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = None
        self.bm25 = None
        self.chunks = []
        self.doc_processor = SECDocumentProcessor()

    def download_10k(self, ticker: str, email: str, num_filings: int = 1):
        """Download SEC 10-K filings for a company"""
        dl = Downloader("FE524-Project", email)
        dl.get("10-K", ticker, limit=num_filings)
        filing_dir = f"sec-edgar-filings/{ticker}/10-K"

        # Find the downloaded filing
        if os.path.exists(filing_dir):
            files = list(Path(filing_dir).rglob("*.txt"))
            if files:
                return str(files[0])
        return None

    def process_and_index_filing(self, filepath: str) -> int:
        """Process SEC filing and create searchable index"""
        # Process the filing
        sections, metadata = self.doc_processor.process_filing(filepath)

        # Create chunks from all sections
        all_chunks = []

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
            # Fallback: If no sections found, chunk the entire document
            st.warning("⚠️ Could not parse document sections. Using full document chunking...")
            raw_content = self.doc_processor.read_filing(filepath)
            clean_text = self.doc_processor.clean_html(raw_content)

            # Create chunks from full text
            words = clean_text.split()
            chunk_size = 1000
            overlap = 200

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

        # Create vector embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

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
            embeddings=embeddings.tolist(),
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

        # Dense retrieval (semantic search)
        query_embedding = self.embedding_model.encode([query])[0]
        dense_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
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
                model="gpt-4o-mini",  # Changed from gpt-4o
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
                model="gpt-4o-mini",  # Changed from gpt-4o
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

    st.title("📊 Financial Filings RAG Analyst Assistant")
    st.markdown("*AI-powered analysis of SEC 10-K filings using RAG*")
    st.markdown("**FE524 Project - Phase 1**")

    # Check for API key in environment
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        st.error("❌ OpenAI API key not found!")
        st.info("""
        Please set your API key as an environment variable:
        
        **Windows Command Prompt:**
        ```
        set OPENAI_API_KEY=your-key-here
        ```
        
        **Windows PowerShell:**
        ```
        $env:OPENAI_API_KEY="your-key-here"
        ```
        
        **Mac/Linux:**
        ```
        export OPENAI_API_KEY=your-key-here
        ```
        
        Then run: `streamlit run financial_rag_app.py` in the same terminal.
        """)
        st.stop()
    else:
        st.success("✅ OpenAI API key detected from environment")

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("🔑 API Key loaded from environment")

        st.header("📁 Document Management")
        ticker = st.text_input("Company Ticker", "AAPL", help="Enter stock ticker symbol (e.g., AAPL, MSFT, TSLA)")
        email = st.text_input("Your Email (for SEC)", "student@stevens.edu", help="Required by SEC Edgar")

        if st.button("📥 Download & Index 10-K", type="primary"):
            with st.spinner("Downloading and processing 10-K filing..."):
                try:
                    # Initialize or get RAG system
                    if 'rag_system' not in st.session_state:
                        st.session_state['rag_system'] = FinancialRAGSystem(api_key)

                    rag_system = st.session_state['rag_system']

                    # Download filing
                    st.info(f"📥 Downloading {ticker} 10-K filing from SEC Edgar...")
                    filepath = rag_system.download_10k(ticker, email)

                    if not filepath:
                        st.error("Failed to download filing. Please check ticker symbol.")
                    else:
                        st.success(f"✅ Downloaded filing: {Path(filepath).name}")

                        # Process and index
                        st.info("🔄 Processing document sections...")
                        num_chunks, metadata = rag_system.process_and_index_filing(filepath)

                        # Store metadata in session
                        st.session_state['metadata'] = metadata
                        st.session_state['ticker'] = ticker

                        st.success(f"✅ Indexed {num_chunks} document chunks")
                        st.balloons()

                        # Display metadata
                        st.markdown("---")
                        st.subheader("📄 Document Info")
                        st.write(f"**Company:** {metadata.get('company_name', ticker)}")
                        st.write(f"**Filing Date:** {metadata.get('filing_date', 'N/A')}")
                        st.write(f"**CIK:** {metadata.get('cik', 'N/A')}")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)

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

    # Main query interface
    st.header("💬 Ask a Question")

    # Get query from session state or text area
    default_query = st.session_state.get('query', '')

    query = st.text_area(
        "Enter your financial analysis question:",
        value=default_query,
        height=100,
        placeholder="e.g., What was the revenue growth rate from 2022 to 2023?",
        key="query_input"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # Process query
    if analyze_btn and query:
        if 'rag_system' not in st.session_state:
            st.warning("⚠️ Please download and index a 10-K filing first using the sidebar.")
        else:
            rag_system = st.session_state['rag_system']

            with st.spinner("🔍 Analyzing documents and generating response..."):
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
                    st.header("📄 Retrieved Document Sections")
                    st.caption("These are the most relevant sections used to generate the answer")

                    for i, chunk in enumerate(result['retrieved_chunks'], 1):
                        with st.expander(
                            f"📑 Source {i}: {chunk['source']} | Relevance: {chunk['relevance_score']:.2%}",
                            expanded=(i == 1)
                        ):
                            st.markdown(chunk['text'])
                            st.caption(f"Company: {chunk['metadata'].get('company', 'N/A')} | Section: {chunk['metadata'].get('section', 'N/A')}")

    # Footer
    st.markdown("---")
    st.caption("FE524-A: Prompt Engineering Lab | Financial RAG Analyst Assistant | Phase 1 Implementation")

if __name__ == "__main__":
    main()
