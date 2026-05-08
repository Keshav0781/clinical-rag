import os
import logging
logging.basicConfig(level=logging.INFO)
import time
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.logger import log_query


# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────
# FastAPI app initialization
# ─────────────────────────────────────────────
app = FastAPI(
    title="ClinicalRAG API",
    description="Intelligent Document Search System for Siemens Healthineers Clinical Knowledge Base",
    version="1.0.0"
)

# ─────────────────────────────────────────────
# CORS Middleware
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"
DOCS_DIR = BASE_DIR / "documents"

# ─────────────────────────────────────────────
# Initialize all models and connections
# Runs ONCE when API starts
# ─────────────────────────────────────────────
print("Initializing ClinicalRAG API...")

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="clinical_docs",
    metadata={"hnsw:space": "cosine"}
)

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading reranker...")
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    USE_CROSS_ENCODER = True
    print("CrossEncoder loaded successfully.")
except Exception as e:
    print(f"CrossEncoder unavailable. Using bi-encoder fallback.")
    USE_CROSS_ENCODER = False

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.2
)

print(f"API initialized. ChromaDB has {collection.count()} chunks.")

# ─────────────────────────────────────────────
# Session Memory Store
# In production this would be Redis or PostgreSQL
# ─────────────────────────────────────────────
session_store = {}


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    n_results: int = 5


class SourceDocument(BaseModel):
    chunk_number: int
    source: str
    page: int
    department: str
    cosine_relevance: float
    rerank_score: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list
    conversation_turn: int
    guardrail_triggered: bool
    guardrail_reason: str
    rerank_method: str
    response_time_seconds: float


class HealthResponse(BaseModel):
    status: str
    total_chunks: int
    rerank_method: str
    groq_model: str
    api_version: str


class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_stored: int


# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def get_biencoder_scores(question: str, chunks: list) -> list:
    question_vec = embedder.encode(question, convert_to_tensor=True)
    chunk_vecs = embedder.encode(chunks, convert_to_tensor=True)
    return [
        float(util.cos_sim(question_vec, chunk_vec)[0][0])
        for chunk_vec in chunk_vecs
    ]


def check_guardrails(question: str, relevance_scores: list, conversation_turn: int = 0) -> dict:
    """
    3 guardrail rules:
    1. Medical advice requests — always blocked regardless of turn
    2. Off-topic questions — always blocked regardless of turn
    3. Low relevance — only blocked on first turn (turn 0)
       Follow-up questions (turn > 0) bypass this rule and rely on conversation memory
    """
    question_lower = question.lower().strip()

    # Rule 1 — Block medical advice requests always
    medical_advice_triggers = [
        "should i", "should my patient", "can i give",
        "is it safe to", "can my patient", "what dose should",
        "recommend for my", "advise me on",
        "tell me if i should", "is this safe for"
    ]
    for trigger in medical_advice_triggers:
        if trigger in question_lower:
            return {
                "is_safe": False,
                "reason": "medical_advice_request",
                "response": (
                    "This system provides information from Siemens Healthineers "
                    "clinical documents only. It is not designed to provide direct "
                    "medical advice or clinical recommendations for specific patients. "
                    "Please consult a qualified physician or clinical specialist."
                )
            }

    # Rule 2 — Block off-topic questions always
    off_topic_triggers = [
        "stock price", "share price", "weather", "football",
        "recipe", "movie", "politics", "election",
        "celebrity", "social media", "cryptocurrency", "bitcoin"
    ]
    for trigger in off_topic_triggers:
        if trigger in question_lower:
            return {
                "is_safe": False,
                "reason": "off_topic",
                "response": (
                    "This system is designed to answer questions about "
                    "Siemens Healthineers clinical documents only. "
                    "Your question appears to be outside the scope of "
                    "the available knowledge base."
                )
            }

    # Rule 3 — Low relevance only blocks on first turn
    # Follow-up questions are intentionally vague and rely on conversation memory
    if relevance_scores and all(score < 0.35 for score in relevance_scores):
        if conversation_turn == 0:
            return {
                "is_safe": False,
                "reason": "low_relevance",
                "response": (
                    "I could not find sufficiently relevant information in the "
                    "available Siemens Healthineers documents to answer your question "
                    "accurately. Please try rephrasing your question or ask about "
                    "a topic covered in the clinical documentation."
                )
            }

    return {"is_safe": True, "reason": "passed", "response": None}


def process_query(user_question: str, session_id: str, n_results: int = 5):
    """
    Core RAG processing function.
    Session-aware — each session_id has its own conversation history.

    Flow:
    1. Get or create session history
    2. Embed question
    3. Search ChromaDB top 20
    4. Rerank — CrossEncoder or bi-encoder fallback
    5. Check guardrails
    6. Detect if retrieval is weak (all rerank scores negative)
    7. Build prompt — clean memory-only prompt if weak, full context prompt if strong
    8. Send to Groq LLM with full conversation history as AIMessage
    9. Save to session
    10. Return result
    """

    # Step 1 — Get or create session history
    if session_id not in session_store:
        session_store[session_id] = []
    conversation_history = session_store[session_id]

    # Step 2 — Embed question
    question_embedding = embedder.encode(user_question).tolist()

    # Step 3 — Fetch top 20 from ChromaDB
    raw_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=20,
        include=["documents", "metadatas", "distances"]
    )

    raw_chunks = raw_results["documents"][0]
    raw_metadatas = raw_results["metadatas"][0]
    raw_distances = raw_results["distances"][0]

    # Step 4 — Reranking with graceful fallback
    if USE_CROSS_ENCODER:
        try:
            rerank_pairs = [[user_question, chunk] for chunk in raw_chunks]
            raw_scores = [float(s) for s in reranker.predict(rerank_pairs)]
            if any(s != s for s in raw_scores):
                raise ValueError("CrossEncoder produced nan scores")
            rerank_scores = raw_scores
            rerank_method = "cross-encoder"
        except Exception:
            rerank_scores = get_biencoder_scores(user_question, raw_chunks)
            rerank_method = "bi-encoder-fallback"
    else:
        rerank_scores = get_biencoder_scores(user_question, raw_chunks)
        rerank_method = "bi-encoder"

    # Sort by rerank score descending
    ranked_results = sorted(
        zip(rerank_scores, raw_chunks, raw_metadatas, raw_distances),
        key=lambda x: x[0],
        reverse=True
    )
    top_results = ranked_results[:n_results]

    # Step 5 — Build context and sources
    context = ""
    sources = []
    relevance_scores = []

    for i, (rerank_score, chunk, meta, distance) in enumerate(top_results):
        cosine_relevance = round(1 - distance, 4)
        relevance_scores.append(cosine_relevance)

        context += (
            f"\n--- Chunk {i+1} "
            f"(Source: {meta['source']}, "
            f"Page: {meta['page']}, "
            f"Department: {meta['department']}) ---\n"
        )
        context += chunk
        context += "\n"

        sources.append({
            "chunk_number": i + 1,
            "source": meta["source"],
            "page": meta["page"],
            "department": meta["department"],
            "cosine_relevance": cosine_relevance,
            "rerank_score": round(rerank_score, 4)
        })

    # Step 6 — Guardrails check
    conversation_turn = len(conversation_history)
    guardrail_result = check_guardrails(user_question, relevance_scores, conversation_turn)
    if not guardrail_result["is_safe"]:
        return {
            "question": user_question,
            "answer": guardrail_result["response"],
            "sources": [],
            "conversation_turn": len(conversation_history) + 1,
            "guardrail_triggered": True,
            "guardrail_reason": guardrail_result["reason"],
            "rerank_method": rerank_method
        }

    # Step 7 — System prompt
    system_prompt = """You are a clinical document assistant at Siemens Healthineers.
Your job is to answer questions based on the provided document context and conversation history.
Always be precise and professional.
If the current question is a follow-up to a previous question in the conversation,
use the conversation history to answer it — do not rely solely on the retrieved chunks.
If the answer cannot be found in either the context or conversation history, respond with:
'I could not find relevant information in the available documents.'
Never make up information that is not in the context or conversation history."""

    # Step 8 — Detect weak retrieval
    # CrossEncoder negative scores mean chunks are irrelevant to the question
    # This happens on follow-up questions like "elaborate on the first point"
    # In this case send a clean memory-only prompt instead of confusing the LLM
    # with irrelevant chunks
    retrieval_is_weak = all(s["rerank_score"] < 0 for s in sources)

    if retrieval_is_weak and conversation_turn > 0:
        # Follow-up with weak retrieval — rely on conversation memory only
        user_prompt = f"""USER QUESTION:
{user_question}

Note: No strongly relevant documents were found for this specific question.
Please answer based on the conversation history above.
If you cannot answer from conversation history, say so clearly."""
    else:
        # Normal query — use retrieved context
        user_prompt = f"""CONTEXT FROM INTERNAL DOCUMENTS:
{context}

USER QUESTION:
{user_question}

Please provide a clear, professional answer based only on the context above.
At the end, mention which documents you used."""

    # Step 9 — Build messages with full conversation history
    # AIMessage tells LLM these are its own previous responses
    # not system instructions — critical for memory to work correctly
    messages = [SystemMessage(content=system_prompt)]
    for turn in conversation_history:
        messages.append(HumanMessage(content=turn["question"]))
        messages.append(AIMessage(content=turn["answer"]))
    messages.append(HumanMessage(content=user_prompt))

    # Step 10 — Call Groq LLM
    response = llm.invoke(messages)

    # Step 11 — Save to session history
    conversation_history.append({
        "question": user_question,
        "answer": response.content
    })
    session_store[session_id] = conversation_history

    return {
        "question": user_question,
        "answer": response.content,
        "sources": sources,
        "conversation_turn": len(conversation_history),
        "guardrail_triggered": False,
        "guardrail_reason": "passed",
        "rerank_method": rerank_method
    }


# ─────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    Used by monitoring systems to verify API is running.
    """
    return {
        "status": "healthy",
        "total_chunks": collection.count(),
        "rerank_method": "cross-encoder" if USE_CROSS_ENCODER else "bi-encoder",
        "groq_model": "llama-3.3-70b-versatile",
        "api_version": "1.0.0"
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Main query endpoint.
    Accepts a question and session_id.
    Returns answer with sources and metadata.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    start_time = time.time()

    result = process_query(
        user_question=request.question,
        session_id=request.session_id,
        n_results=request.n_results
    )

    end_time = time.time()
    response_time = round(end_time - start_time, 3)

    # Log every query to BigQuery for audit trail and Power BI dashboard
    log_query(
        session_id=request.session_id,
        question=result["question"],
        answer=result["answer"],
        sources=result["sources"],
        response_time=response_time,
        guardrail_triggered=result["guardrail_triggered"],
        guardrail_reason=result["guardrail_reason"],
        rerank_method=result["rerank_method"],
        conversation_turn=result["conversation_turn"]
    )

    return {
        "question": result["question"],
        "answer": result["answer"],
        "sources": result["sources"],
        "conversation_turn": result["conversation_turn"],
        "guardrail_triggered": result["guardrail_triggered"],
        "guardrail_reason": result["guardrail_reason"],
        "rerank_method": result["rerank_method"],
        "response_time_seconds": response_time
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents():
    """
    Document ingestion endpoint.
    Triggers re-ingestion of all PDFs in documents/ folder.
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from datetime import date
        from tqdm import tqdm

        pdf_files = list(DOCS_DIR.glob("*.pdf"))
        if not pdf_files:
            raise HTTPException(
                status_code=404,
                detail="No PDF files found in documents/ folder"
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        def is_valid_chunk(text: str) -> bool:
            cleaned = text.strip()
            if len(cleaned) < 50:
                return False
            if len(cleaned.split()) < 10:
                return False
            noise_patterns = [
                "table of contents", "all rights reserved",
                "©", "siemens healthineers", "www.", "http",
                "page ", "figure ", "references"
            ]
            lower_text = cleaned.lower()
            if any(lower_text.startswith(p) for p in noise_patterns):
                if len(cleaned) < 100:
                    return False
            return True

        def detect_department(filename: str) -> str:
            f = filename.lower()
            if any(t in f for t in ["spect", "xspect", "iqspect", "mi_", "_mi"]):
                return "Molecular Imaging"
            elif any(t in f for t in ["oncology", "cancer", "tumor"]):
                return "Oncology"
            elif any(t in f for t in ["operational", "excellence", "operations"]):
                return "Operations"
            elif any(t in f for t in ["rad", "radiology", "imaging"]):
                return "Radiology Services"
            return "General Clinical"

        total_stored = 0
        for pdf_path in pdf_files:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            chunks = text_splitter.split_documents(pages)
            department = detect_department(pdf_path.name)

            for i, chunk in enumerate(chunks):
                if not is_valid_chunk(chunk.page_content):
                    continue
                chunk_id = f"{pdf_path.stem}_{i}"
                embedding = embedder.encode(chunk.page_content).tolist()
                collection.upsert(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk.page_content],
                    metadatas=[{
                        "source": pdf_path.name,
                        "page": chunk.metadata.get("page", 0),
                        "document_type": "Clinical White Paper",
                        "department": department,
                        "date_processed": str(date.today()),
                        "word_count": len(chunk.page_content.split())
                    }]
                )
                total_stored += 1

        return {
            "status": "success",
            "message": f"Successfully ingested {len(pdf_files)} documents",
            "chunks_stored": total_stored
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    Clear conversation history for a specific session.
    """
    if session_id in session_store:
        session_store.pop(session_id)
        return {"status": "success", "message": f"Session {session_id} cleared"}
    return {"status": "not_found", "message": f"Session {session_id} not found"}