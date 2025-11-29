"""
Financial RAG Analyst - AI-powered Q&A for SEC 10-K filings

Setup:
    uv venv && .venv\\Scripts\\activate && uv pip install -r requirements.txt
    Create .env with: OPENAI_API_KEY=your-key-here
    Run: streamlit run financial_rag.py

Required API access: gpt-5-mini, text-embedding-3-small
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


# ============== SPECIALIZED AGENTS ==============

class FinancialAnalystAgent:
    """Specialized agent for financial analysis and calculations"""
    
    def __init__(self, client):
        self.client = client
        self.name = "Financial Analyst"
        self.system_prompt = """You are a financial analyst expert. Your role is to:
- Extract and analyze financial metrics (revenue, profit, expenses)
- Calculate financial ratios (P/E, debt-to-equity, operating margin)
- Identify trends and patterns in financial data
- Provide quantitative insights with calculations shown"""
    
    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "extract_metric",
                    "description": "Extract a specific financial metric from the document context",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metric_name": {"type": "string", "description": "Name of metric (revenue, net_income, operating_expenses, etc.)"},
                            "value": {"type": "number", "description": "The extracted numeric value"},
                            "unit": {"type": "string", "description": "Unit (millions, billions, percentage)"},
                            "source": {"type": "string", "description": "Where this was found in the document"}
                        },
                        "required": ["metric_name", "value"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_ratio",
                    "description": "Calculate a financial ratio",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ratio_name": {"type": "string", "description": "Name of the ratio"},
                            "formula": {"type": "string", "description": "Formula used"},
                            "numerator": {"type": "number"},
                            "denominator": {"type": "number"},
                            "result": {"type": "number"}
                        },
                        "required": ["ratio_name", "result"]
                    }
                }
            }
        ]
    
    def analyze(self, question: str, context: str) -> Dict:
        """Perform financial analysis"""
        prompt = f"""Analyze the following financial data to answer the question.

Context from SEC 10-K filing:
{context}

Question: {question}

Extract relevant metrics, perform calculations, and provide a detailed financial analysis.
Use the tools to extract metrics and calculate ratios where appropriate."""

        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            tools=self.get_tools(),
            tool_choice="auto"
        )
        
        return self._process_response(response, question, context)
    
    def _process_response(self, response, question: str, context: str) -> Dict:
        """Process agent response and tool calls"""
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        
        extracted_data = []
        calculations = []
        
        for tool_call in tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
                if tool_call.function.name == "extract_metric":
                    extracted_data.append(args)
                elif tool_call.function.name == "calculate_ratio":
                    calculations.append(args)
            except:
                pass
        
        # Generate final analysis
        final_prompt = f"""Based on your analysis:
Question: {question}
Extracted Data: {json.dumps(extracted_data)}
Calculations: {json.dumps(calculations)}

Provide a clear, professional answer with citations."""

        final_response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": final_prompt}
            ]
        )
        
        return {
            "answer": final_response.choices[0].message.content,
            "extracted_data": extracted_data,
            "calculations": calculations,
            "agent": self.name
        }


class ComplianceAgent:
    """Specialized agent for compliance checking"""
    
    def __init__(self, client):
        self.client = client
        self.name = "Compliance Officer"
        self.system_prompt = """You are a compliance officer expert. Your role is to:
- Check SEC regulatory compliance
- Verify disclosure requirements are met
- Identify missing or incomplete information
- Flag potential compliance issues"""
    
    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "check_disclosure",
                    "description": "Check if a required disclosure is present",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "disclosure_type": {"type": "string"},
                            "is_present": {"type": "boolean"},
                            "completeness": {"type": "string", "enum": ["complete", "partial", "missing"]},
                            "notes": {"type": "string"}
                        },
                        "required": ["disclosure_type", "is_present"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "flag_issue",
                    "description": "Flag a potential compliance issue",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "issue_type": {"type": "string"},
                            "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                            "description": {"type": "string"},
                            "recommendation": {"type": "string"}
                        },
                        "required": ["issue_type", "severity", "description"]
                    }
                }
            }
        ]
    
    def analyze(self, question: str, context: str) -> Dict:
        """Perform compliance check"""
        prompt = f"""Review the following SEC filing content for compliance.

Context from SEC 10-K filing:
{context}

Question: {question}

Check for required disclosures and flag any compliance issues."""

        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            tools=self.get_tools(),
            tool_choice="auto"
        )
        
        return self._process_response(response, question)
    
    def _process_response(self, response, question: str) -> Dict:
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        
        disclosures = []
        issues = []
        
        for tool_call in tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
                if tool_call.function.name == "check_disclosure":
                    disclosures.append(args)
                elif tool_call.function.name == "flag_issue":
                    issues.append(args)
            except:
                pass
        
        final_prompt = f"""Based on your compliance review:
Question: {question}
Disclosures Checked: {json.dumps(disclosures)}
Issues Found: {json.dumps(issues)}

Provide a compliance assessment summary."""

        final_response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": final_prompt}
            ]
        )
        
        return {
            "answer": final_response.choices[0].message.content,
            "disclosures": disclosures,
            "issues": issues,
            "agent": self.name
        }


class RiskAssessmentAgent:
    """Specialized agent for risk analysis"""
    
    def __init__(self, client):
        self.client = client
        self.name = "Risk Analyst"
        self.system_prompt = """You are a risk assessment expert. Your role is to:
- Identify and categorize business risks
- Assess risk severity and likelihood
- Compare risks across categories
- Provide risk mitigation insights"""
    
    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "identify_risk",
                    "description": "Identify a specific risk factor",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "risk_name": {"type": "string"},
                            "category": {"type": "string", "enum": ["financial", "operational", "regulatory", "market", "cybersecurity", "environmental", "competitive"]},
                            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                            "description": {"type": "string"}
                        },
                        "required": ["risk_name", "category", "severity"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "assess_impact",
                    "description": "Assess the potential impact of a risk",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "risk_name": {"type": "string"},
                            "financial_impact": {"type": "string"},
                            "likelihood": {"type": "string", "enum": ["unlikely", "possible", "likely", "very_likely"]},
                            "mitigation": {"type": "string"}
                        },
                        "required": ["risk_name", "likelihood"]
                    }
                }
            }
        ]
    
    def analyze(self, question: str, context: str) -> Dict:
        """Perform risk assessment"""
        prompt = f"""Analyze the following SEC filing for risk factors.

Context from SEC 10-K filing:
{context}

Question: {question}

Identify risks, assess their severity, and provide insights."""

        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            tools=self.get_tools(),
            tool_choice="auto"
        )
        
        return self._process_response(response, question)
    
    def _process_response(self, response, question: str) -> Dict:
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        
        risks = []
        impacts = []
        
        for tool_call in tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
                if tool_call.function.name == "identify_risk":
                    risks.append(args)
                elif tool_call.function.name == "assess_impact":
                    impacts.append(args)
            except:
                pass
        
        final_prompt = f"""Based on your risk assessment:
Question: {question}
Identified Risks: {json.dumps(risks)}
Impact Assessments: {json.dumps(impacts)}

Provide a comprehensive risk analysis summary."""

        final_response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": final_prompt}
            ]
        )
        
        return {
            "answer": final_response.choices[0].message.content,
            "risks": risks,
            "impacts": impacts,
            "agent": self.name
        }


class AgentRouter:
    """Routes questions to appropriate specialized agents"""
    
    # Keywords that trigger each agent
    ANALYST_KEYWORDS = ["revenue", "profit", "margin", "ratio", "calculate", "growth", "earnings", 
                        "expense", "income", "eps", "ebitda", "cash flow", "financial", "metric"]
    COMPLIANCE_KEYWORDS = ["compliance", "regulation", "disclosure", "sec", "audit", "requirement",
                          "filing", "report", "statement", "governance", "control"]
    RISK_KEYWORDS = ["risk", "threat", "challenge", "uncertainty", "exposure", "vulnerability",
                    "cybersecurity", "competition", "regulatory risk", "market risk"]
    
    def __init__(self, client):
        self.client = client
        self.analyst_agent = FinancialAnalystAgent(client)
        self.compliance_agent = ComplianceAgent(client)
        self.risk_agent = RiskAssessmentAgent(client)
    
    def detect_agent_need(self, question: str) -> tuple:
        """Detect which agent(s) should handle the question"""
        question_lower = question.lower()
        
        scores = {
            "analyst": sum(1 for kw in self.ANALYST_KEYWORDS if kw in question_lower),
            "compliance": sum(1 for kw in self.COMPLIANCE_KEYWORDS if kw in question_lower),
            "risk": sum(1 for kw in self.RISK_KEYWORDS if kw in question_lower)
        }
        
        # Determine if agents are needed and which one
        max_score = max(scores.values())
        if max_score == 0:
            return False, None  # Use standard RAG
        
        # Return the agent with highest score
        primary_agent = max(scores, key=scores.get)
        return True, primary_agent
    
    def route_and_execute(self, question: str, context: str) -> Dict:
        """Route question to appropriate agent and execute"""
        needs_agent, agent_type = self.detect_agent_need(question)
        
        if not needs_agent:
            return None  # Signal to use standard RAG
        
        # Execute the appropriate agent
        if agent_type == "analyst":
            result = self.analyst_agent.analyze(question, context)
        elif agent_type == "compliance":
            result = self.compliance_agent.analyze(question, context)
        elif agent_type == "risk":
            result = self.risk_agent.analyze(question, context)
        else:
            return None
        
        result["agent_type"] = agent_type
        return result


# ============== MAIN RAG SYSTEM ==============

class FinancialRAGSystem:
    """Main RAG system for financial document analysis with hybrid RAG + Agent support"""

    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        # Using OpenAI's text-embedding-3-small for better semantic search
        self.embedding_model_name = "text-embedding-3-small"
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = None
        self.bm25 = None
        self.chunks = []
        self.doc_processor = SECDocumentProcessor()
        # Initialize agent router for hybrid mode
        self.agent_router = AgentRouter(self.client)

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
                model="gpt-5-mini",  # Using GPT-5-mini for advanced financial analysis
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert at extracting data and performing calculations."},
                    {"role": "user", "content": calculation_prompt}
                ]
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
                model="gpt-5-mini",  # Using GPT-5-mini for advanced financial analysis
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst providing evidence-based insights."},
                    {"role": "user", "content": answer_prompt}
                ]
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

    def query(self, question: str, force_mode: str = None) -> Dict[str, Any]:
        """Main query interface with hybrid RAG + Agent support
        
        Args:
            question: The user's question
            force_mode: Optional - 'rag' for RAG-only, 'agent' for agent-only, None for auto-detect
        
        Returns:
            Dict with answer, sources, and metadata about which mode was used
        """
        if not self.chunks:
            return {"error": "No documents indexed. Please load documents first."}

        # Retrieve relevant chunks (always needed for context)
        retrieved = self.hybrid_retrieve(question, top_k=5)
        context = "\n\n".join([
            f"[Source {i+1}: {chunk['source']}]\n{chunk['text']}"
            for i, chunk in enumerate(retrieved)
        ])

        # Determine which mode to use
        if force_mode == 'rag':
            use_agents = False
        elif force_mode == 'agent':
            use_agents = True
        else:
            # Auto-detect based on question
            use_agents, _ = self.agent_router.detect_agent_need(question)

        if use_agents:
            # Use agent-based approach
            agent_result = self.agent_router.route_and_execute(question, context)
            
            if agent_result:
                return {
                    'answer': agent_result['answer'],
                    'retrieved_chunks': retrieved,
                    'metrics': {},
                    'calculations': agent_result.get('calculations', []),
                    'insights': [],
                    'mode': 'agent',
                    'agent_used': agent_result.get('agent', 'Unknown'),
                    'agent_data': {
                        'extracted_data': agent_result.get('extracted_data', []),
                        'risks': agent_result.get('risks', []),
                        'disclosures': agent_result.get('disclosures', []),
                        'issues': agent_result.get('issues', [])
                    }
                }
        
        # Use standard RAG approach
        result = self.generate_answer(question, retrieved)
        result['mode'] = 'rag'
        result['agent_used'] = None
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
                    # Show which mode was used
                    mode = result.get('mode', 'rag')
                    agent_used = result.get('agent_used')
                    
                    if mode == 'agent' and agent_used:
                        st.success(f"🤖 **Agent Mode:** {agent_used}")
                    else:
                        st.info("📚 **RAG Mode:** Direct retrieval and generation")
                    
                    # Display answer
                    st.header("📝 Answer")
                    st.markdown(result['answer'])

                    # Display agent-specific data if available
                    agent_data = result.get('agent_data', {})
                    
                    # Display extracted financial data (from analyst agent)
                    if agent_data.get('extracted_data'):
                        st.header("📊 Extracted Financial Data")
                        for item in agent_data['extracted_data']:
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                st.write(f"**{item.get('metric_name', 'Metric')}**")
                            with col2:
                                value = item.get('value', 'N/A')
                                unit = item.get('unit', '')
                                st.write(f"{value} {unit}")
                    
                    # Display risks (from risk agent)
                    if agent_data.get('risks'):
                        st.header("⚠️ Identified Risks")
                        for risk in agent_data['risks']:
                            severity = risk.get('severity', 'unknown')
                            severity_color = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}.get(severity, '⚪')
                            st.write(f"{severity_color} **{risk.get('risk_name', 'Risk')}** ({risk.get('category', 'General')})")
                            st.caption(risk.get('description', ''))
                    
                    # Display compliance issues (from compliance agent)
                    if agent_data.get('issues'):
                        st.header("🔍 Compliance Issues")
                        for issue in agent_data['issues']:
                            severity = issue.get('severity', 'unknown')
                            severity_color = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(severity, '⚪')
                            st.write(f"{severity_color} **{issue.get('issue_type', 'Issue')}**")
                            st.caption(issue.get('description', ''))

                    # Display metrics (from RAG mode)
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
                            if isinstance(calc, dict):
                                st.write(f"**{calc.get('ratio_name', 'Calculation')}:** {calc.get('result', 'N/A')}")
                                if calc.get('formula'):
                                    st.caption(f"Formula: {calc['formula']}")
                            else:
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
