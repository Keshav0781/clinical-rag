import os
import logging
logging.basicConfig(level=logging.INFO)
import time
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
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
    department_filter: str = "all"


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
    follow_up_questions: list = []


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
def generate_follow_up_questions(question: str, answer: str) -> list:
    """
    Generates 3 suggested follow-up questions based on the answer.
    Uses a separate lightweight LLM call.
    Returns empty list on any failure — never blocks main response.
    """
    try:
        prompt = f"""You are analyzing a conversation about Siemens Healthineers clinical documents.

Question: {question}
Answer: {answer[:500]}

First decide: does this answer contain substantive clinical or technical information worth exploring further?
If the answer is conversational (like done, thanks, ok, goodbye) or says it could not find information, respond with exactly: NONE

If the answer contains real clinical information, generate exactly 3 short follow-up questions.
Rules:
- Each question must be specific and relevant to the clinical content in the answer
- Keep each question under 12 words
- Return only the 3 questions, one per line, no numbering, no bullets
- Do not include any other text"""

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        if content.upper() == 'NONE' or content.upper().startswith('NONE'):
            return []
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        return lines[:3]
    except Exception:
        return []



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
    if relevance_scores and all(score < 0.30 for score in relevance_scores):
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


def process_query(user_question: str, session_id: str, n_results: int = 5, department_filter: str = "all"):
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
    where_filter = {"department": {"$eq": department_filter}} if department_filter and department_filter != "all" else None
    raw_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=20,
        include=["documents", "metadatas", "distances"],
        where=where_filter
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


@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    """
    Serves the professional chat interface at the root URL.
    Anyone visiting the base URL sees this instead of a blank page.
    Technical users can still access /docs for Swagger UI.
    """
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ClinicalRAG</title>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0c10;
            --bg-secondary: #0f1218;
            --bg-tertiary: #151820;
            --bg-hover: #1a1e28;
            --border: #1e2330;
            --border-light: #252a38;
            --accent: #00c4b4;
            --accent-dim: #00c4b414;
            --accent-hover: #00d4c2;
            --text-primary: #e8eaf0;
            --text-secondary: #8892a4;
            --text-muted: #4a5268;
            --user-bubble: #0e3a5c;
            --user-border: #1a5a8a;
            --error: #ff4d6d;
            --error-bg: #1a0a0f;
            --sidebar-width: 260px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'DM Sans', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            overflow: hidden;
        }

        .sidebar {
            width: var(--sidebar-width);
            background: var(--bg-secondary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }

        .sidebar-header {
            padding: 20px 16px 16px;
            border-bottom: 1px solid var(--border);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 16px;
        }

        .brand-icon {
            width: 32px;
            height: 32px;
            background: var(--accent);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'DM Mono', monospace;
            font-size: 11px;
            font-weight: 500;
            color: #000;
            letter-spacing: 0.5px;
        }

        .brand-name {
            font-size: 15px;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.3px;
        }

        .new-chat-btn {
            width: 100%;
            background: var(--accent-dim);
            border: 1px solid rgba(0,196,180,0.3);
            color: var(--accent);
            padding: 9px 14px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            font-family: 'DM Sans', sans-serif;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s;
        }

        .new-chat-btn:hover { background: var(--accent); color: #000; }
        .new-chat-btn svg { width: 14px; height: 14px; }

        .sidebar-section-label {
            font-size: 10px;
            font-weight: 500;
            color: var(--text-muted);
            letter-spacing: 1px;
            text-transform: uppercase;
            padding: 16px 16px 8px;
        }

        .history-list {
            flex: 1;
            overflow-y: auto;
            padding: 4px 8px;
        }

        .history-item {
            padding: 9px 10px;
            border-radius: 7px;
            cursor: pointer;
            font-size: 13px;
            color: var(--text-secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: all 0.15s;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .history-item:hover { background: var(--bg-hover); color: var(--text-primary); }
        .history-item.active { background: var(--bg-hover); color: var(--text-primary); }
        .history-item svg { width: 13px; height: 13px; flex-shrink: 0; opacity: 0.5; }

        .history-title {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .rename-input {
            flex: 1;
            background: var(--bg-primary);
            border: 1px solid var(--accent);
            border-radius: 4px;
            color: var(--text-primary);
            font-size: 13px;
            font-family: 'DM Sans', sans-serif;
            padding: 2px 6px;
            outline: none;
            width: 100%;
        }
        .rename-btn {
            opacity: 0;
            font-size: 13px;
            color: var(--text-muted);
            cursor: pointer;
            padding: 0 2px;
            transition: opacity 0.2s;
            flex-shrink: 0;
        }

        .history-item:hover .rename-btn {
            opacity: 1;
        }

        .history-empty {
            padding: 12px 16px;
            font-size: 12px;
            color: var(--text-muted);
            font-style: italic;
        }

        .sidebar-footer {
            padding: 12px 16px;
            border-top: 1px solid var(--border);
            font-size: 11px;
            color: var(--text-muted);
            line-height: 1.6;
        }

        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        .topbar {
            padding: 14px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .topbar-title {
            font-size: 14px;
            font-weight: 500;
            color: var(--text-secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 400px;
        }

        .topbar-status {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            color: var(--text-muted);
            flex-shrink: 0;
        }

        .topbar-right {
            display: flex;
            align-items: center;
            gap: 12px;
            flex-shrink: 0;
        }

        .dept-filter {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-light);
            border-radius: 6px;
            color: var(--text-secondary);
            font-size: 12px;
            font-family: 'DM Sans', sans-serif;
            padding: 5px 10px;
            cursor: pointer;
            outline: none;
            transition: border-color 0.2s;
        }

        .dept-filter:hover { border-color: var(--accent); }
        .dept-filter:focus { border-color: var(--accent); }

        .status-dot {
            width: 7px;
            height: 7px;
            background: var(--accent);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }

        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 32px 24px;
        }

        .chat-inner {
            max-width: 720px;
            margin: 0 auto;
        }

        .welcome {
            padding: 48px 0 32px;
            text-align: center;
        }

        .welcome-icon {
            width: 56px;
            height: 56px;
            background: var(--accent-dim);
            border: 1px solid rgba(0,196,180,0.3);
            border-radius: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            font-family: 'DM Mono', monospace;
            font-size: 14px;
            font-weight: 500;
            color: var(--accent);
        }

        .welcome h2 {
            font-size: 22px;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.5px;
            margin-bottom: 10px;
        }

        .welcome p {
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.6;
            max-width: 420px;
            margin: 0 auto 32px;
        }

        .example-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            max-width: 580px;
            margin: 0 auto;
        }

        .example-card {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-light);
            border-radius: 10px;
            padding: 14px 16px;
            cursor: pointer;
            text-align: left;
            font-size: 13px;
            font-family: 'DM Sans', sans-serif;
            color: var(--text-secondary);
            line-height: 1.5;
            transition: all 0.2s;
        }

        .example-card:hover {
            border-color: var(--accent);
            color: var(--text-primary);
            background: var(--bg-hover);
            transform: translateY(-1px);
        }

        .message {
            margin-bottom: 28px;
            animation: slideUp 0.25s ease;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .msg-user {
            display: flex;
            justify-content: flex-end;
        }

        .msg-user .bubble {
            background: var(--user-bubble);
            border: 1px solid var(--user-border);
            color: #c8e6ff;
            padding: 12px 16px;
            border-radius: 14px 14px 3px 14px;
            max-width: 65%;
            font-size: 14px;
            line-height: 1.6;
        }

        .msg-assistant {
            display: flex;
            gap: 12px;
            align-items: flex-start;
        }

        .avatar {
            width: 30px;
            height: 30px;
            background: var(--accent-dim);
            border: 1px solid rgba(0,196,180,0.3);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            font-weight: 500;
            color: var(--accent);
            flex-shrink: 0;
            margin-top: 2px;
        }

        .msg-content { flex: 1; }

        .bubble-assistant {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-light);
            padding: 14px 18px;
            border-radius: 3px 14px 14px 14px;
            font-size: 14px;
            line-height: 1.75;
            color: var(--text-primary);
        }

        .bubble-guardrail {
            background: var(--error-bg);
            border-color: #3a1520;
            color: #ff8fa3;
        }

        .sources-row {
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }

        .source-chip {
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 5px;
            padding: 4px 9px;
            font-size: 11px;
            font-family: 'DM Mono', monospace;
            color: var(--text-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 240px;
        }


        .msg-meta {
            margin-top: 8px;
            display: flex;
            gap: 14px;
            font-size: 11px;
            color: var(--text-muted);
            font-family: 'DM Mono', monospace;
        }

        .followup-row {
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }


        .followup-chip {
            background: var(--accent-dim);
            border: 1px solid rgba(0,196,180,0.3);
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 12px;
            font-family: 'DM Sans', sans-serif;
            color: var(--accent);
            cursor: pointer;
            transition: all 0.2s;
            text-align: left;
            line-height: 1.4;
        }

        .followup-chip:hover {
            background: var(--accent);
            color: #000;
        }


        .meta-guardrail { color: var(--error); }

        .loading-row {
            display: flex;
            gap: 12px;
            align-items: flex-start;
            margin-bottom: 28px;
        }

        .typing {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-light);
            padding: 16px 18px;
            border-radius: 3px 14px 14px 14px;
            display: flex;
            gap: 5px;
            align-items: center;
        }

        .typing span {
            width: 6px;
            height: 6px;
            background: var(--accent);
            border-radius: 50%;
            animation: typingBounce 1.3s infinite;
        }

        .typing span:nth-child(2) { animation-delay: 0.15s; }
        .typing span:nth-child(3) { animation-delay: 0.3s; }

        @keyframes typingBounce {
            0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
            30% { transform: translateY(-5px); opacity: 1; }
        }

        .input-bar {
            padding: 16px 24px 20px;
            border-top: 1px solid var(--border);
        }

        .input-inner { max-width: 720px; margin: 0 auto; }

        .input-box {
            display: flex;
            gap: 10px;
            align-items: flex-end;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-light);
            border-radius: 12px;
            padding: 10px 10px 10px 16px;
            transition: border-color 0.2s;
        }

        .input-box:focus-within { border-color: var(--accent); }

        textarea {
            flex: 1;
            background: transparent;
            border: none;
            outline: none;
            color: var(--text-primary);
            font-size: 14px;
            font-family: 'DM Sans', sans-serif;
            resize: none;
            min-height: 24px;
            max-height: 120px;
            line-height: 1.6;
            padding: 2px 0;
        }

        textarea::placeholder { color: var(--text-muted); }

        .send-btn {
            width: 36px;
            height: 36px;
            background: var(--accent);
            border: none;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            transition: all 0.2s;
        }

        .send-btn:hover { background: var(--accent-hover); transform: scale(1.05); }
        .send-btn:disabled { background: var(--bg-hover); cursor: not-allowed; transform: none; }
        .send-btn svg { width: 16px; height: 16px; fill: #000; }
        .send-btn:disabled svg { fill: var(--text-muted); }

        .input-hint {
            margin-top: 8px;
            font-size: 11px;
            color: var(--text-muted);
            text-align: center;
        }

        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 3px; }

        @media (max-width: 640px) {
            .sidebar { display: none; }
            .example-grid { grid-template-columns: 1fr; }
            .msg-user .bubble { max-width: 85%; }
        }
    </style>
</head>
<body>

<aside class="sidebar">
    <div class="sidebar-header">
        <div class="brand">
            <div class="brand-icon">CR</div>
            <span class="brand-name">ClinicalRAG</span>
        </div>
        <button class="new-chat-btn" onclick="newChat()">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
            New conversation
        </button>
    </div>
    <div class="sidebar-section-label">Recent</div>
    <div class="history-list" id="historyList"></div>
    <div class="sidebar-footer">
        Groq Llama 3.3 70B · ChromaDB<br>GCP Cloud Run · europe-west1
    </div>
</aside>

<main class="main">
    <div class="topbar">
        <span class="topbar-title" id="topbarTitle">New conversation</span>
        <div class="topbar-right">
            <select class="dept-filter" id="deptFilter" onchange="updateDeptFilter()">
                <option value="all">All departments</option>
                <option value="Molecular Imaging">Molecular Imaging</option>
                <option value="Oncology">Oncology</option>
                <option value="Operations">Operations</option>
                <option value="Radiology Services">Radiology Services</option>
            </select>
            <div class="topbar-status">
                <div class="status-dot"></div>
                <span>834 chunks · cross-encoder</span>
            </div>
        </div>
    </div>

    <div class="chat-area" id="chatArea">
        <div class="chat-inner" id="chatInner"></div>
    </div>

    <div class="input-bar">
        <div class="input-inner">
            <div class="input-box">
                <textarea id="questionInput"
                    placeholder="Ask a question about Siemens Healthineers clinical documents..."
                    rows="1"
                    onkeydown="handleKey(event)"
                    oninput="resize(this)"></textarea>
                <button class="send-btn" id="sendBtn" onclick="send()">
                    <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
                </button>
            </div>
            <div class="input-hint">Enter to send · Shift+Enter for new line · Ask complete questions for best results</div>
        </div>
    </div>
</main>

<script>
const STORAGE_KEY = 'clinicalrag_v1';
let sessions = [];
try { sessions = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]'); } catch(e) { sessions = []; }
let currentId = null;
let currentMessages = [];
let loading = false;

function save() {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions)); } catch(e) {}
}

function renderHistory() {
    const list = document.getElementById('historyList');
    if (!sessions.length) {
        list.innerHTML = '<div class="history-empty">No conversations yet</div>';
        return;
    }
    list.innerHTML = sessions.slice().reverse().map(s =>
        `<div class="history-item ${s.id === currentId ? 'active' : ''}" onclick="loadSession('${s.id}')">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            <span class="history-title">${esc(s.title)}</span>
            <span class="rename-btn" onclick="event.stopPropagation();renameSession('${s.id}')" title="Rename">✎</span>
        </div>`
    ).join('');
}

function renameSession(id) {
    const session = sessions.find(s => s.id === id);
    if (!session) return;
    const newTitle = prompt('Rename conversation:', session.title);
    if (newTitle && newTitle.trim()) {
        session.title = newTitle.trim();
        save();
        if (currentId === id) {
            document.getElementById('topbarTitle').textContent = newTitle.trim();
        }
        renderHistory();
    }
}

function showWelcome() {
    document.getElementById('chatInner').innerHTML = `
        <div class="welcome" id="welcomeScreen">
            <div class="welcome-icon">CR</div>
            <h2>Ask anything about clinical documents</h2>
            <p>Search across Siemens Healthineers clinical white papers using natural language. Get precise answers with source references. For best results, ask complete questions — for example, "What does xSPECT stand for?" rather than "xSPECT full form".</p>
            <div class="example-grid">
                <button class="example-card" onclick="useExample(this)">What is xSPECT Bone and what are its clinical benefits?</button>
                <button class="example-card" onclick="useExample(this)">How does iQSPECT improve cardiac imaging?</button>
                <button class="example-card" onclick="useExample(this)">What oncology solutions does Siemens offer?</button>
                <button class="example-card" onclick="useExample(this)">What are the operational excellence benefits for healthcare providers?</button>
            </div>
        </div>`;
}

function newChat() {
    currentId = 'session-' + Date.now();
    currentMessages = [];
    document.getElementById('topbarTitle').textContent = 'New conversation';
    document.getElementById('questionInput').value = '';
    showWelcome();
    renderHistory();
}

function loadSession(id) {
    const s = sessions.find(x => x.id === id);
    if (!s) return;
    currentId = id;
    currentMessages = s.messages || [];
    document.getElementById('topbarTitle').textContent = s.title;
    document.getElementById('chatInner').innerHTML = '';
    currentMessages.forEach(m => {
        if (m.role === 'user') addUserBubble(m.text, false);
        else addAssistantBubble(m.data, false);
    });
    renderHistory();
    scrollDown();
}

function useExample(btn) {
    const input = document.getElementById('questionInput');
    input.value = btn.textContent.trim();
    resize(input);
    input.focus();
}

function useFollowUp(btn) {
    const q = btn.textContent.trim();
    document.getElementById('questionInput').value = q;
    resize(document.getElementById('questionInput'));
    send();
}


let currentDeptFilter = 'all';

function updateDeptFilter() {
    currentDeptFilter = document.getElementById('deptFilter').value;
}

function resize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
}

function removeWelcome() {
    const w = document.getElementById('welcomeScreen');
    if (w) w.remove();
}

function addUserBubble(text, store = true) {
    removeWelcome();
    const inner = document.getElementById('chatInner');
    const d = document.createElement('div');
    d.className = 'message msg-user';
    d.innerHTML = `<div class="bubble">${esc(text)}</div>`;
    inner.appendChild(d);
    if (store) currentMessages.push({ role: 'user', text });
    scrollDown();
}

function showTyping() {
    const inner = document.getElementById('chatInner');
    const d = document.createElement('div');
    d.className = 'loading-row'; d.id = 'typing';
    d.innerHTML = `<div class="avatar">CR</div><div class="typing"><span></span><span></span><span></span></div>`;
    inner.appendChild(d);
    scrollDown();
}

function hideTyping() {
    const t = document.getElementById('typing');
    if (t) t.remove();
}

function addAssistantBubble(data, store = true) {
    const inner = document.getElementById('chatInner');
    const d = document.createElement('div');
    d.className = 'message msg-assistant';
    const guard = data.guardrail_triggered;
    const uniq = data.sources ? [...new Set(data.sources.map(s => s.source))] : [];
    const chips = uniq.map(s =>
        `<span class="source-chip" title="${esc(s)}">${esc(s.replace('.pdf',''))}</span>`
    ).join('');
    const followUps = (data.follow_up_questions && data.follow_up_questions.length > 0 && !guard)
        ? `<div class="followup-row">${data.follow_up_questions.map(q =>
            `<button class="followup-chip" onclick="useFollowUp(this)">${esc(q)}</button>`
          ).join('')}</div>`
        : '';
    d.innerHTML = `
        <div class="avatar">CR</div>
        <div class="msg-content">
            <div class="${guard ? 'bubble-assistant bubble-guardrail' : 'bubble-assistant'}">${esc(data.answer)}</div>
            ${chips ? `<div class="sources-row">${chips}</div>` : ''}
            <div class="msg-meta">
                <span>${data.response_time_seconds}s</span>
                <span>turn ${data.conversation_turn}</span>
                ${guard ? `<span class="meta-guardrail">guardrail: ${esc(data.guardrail_reason)}</span>` : `<span>${esc(data.rerank_method)}</span>`}
            </div>
            ${followUps}
        </div>`;
    inner.appendChild(d);
    if (store) currentMessages.push({ role: 'assistant', data });
    scrollDown();
}

function scrollDown() {
    const a = document.getElementById('chatArea');
    a.scrollTop = a.scrollHeight;
}

function esc(str) {
    const d = document.createElement('div');
    d.appendChild(document.createTextNode(str || ''));
    return d.innerHTML;
}

async function send() {
    const input = document.getElementById('questionInput');
    const q = input.value.trim();
    if (!q || loading) return;
    if (!currentId) { currentId = 'session-' + Date.now(); currentMessages = []; }

    loading = true;
    document.getElementById('sendBtn').disabled = true;
    input.value = ''; input.style.height = 'auto';

    addUserBubble(q);
    showTyping();

    try {
        const res = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q, session_id: currentId, department_filter: currentDeptFilter })
        });
        const data = await res.json();
        hideTyping();
        addAssistantBubble(data);

        const title = q.length > 45 ? q.slice(0, 45) + '...' : q;
        const existing = sessions.find(s => s.id === currentId);
        if (existing) { existing.messages = currentMessages; }
        else {
            sessions.push({ id: currentId, title, messages: currentMessages });
            document.getElementById('topbarTitle').textContent = title;
        }
        save();
        renderHistory();
    } catch(err) {
        hideTyping();
        addAssistantBubble({
            answer: 'Something went wrong. Please try again.',
            guardrail_triggered: true,
            guardrail_reason: 'error',
            sources: [],
            response_time_seconds: 0,
            conversation_turn: 0,
            rerank_method: ''
        }, false);
    }

    loading = false;
    document.getElementById('sendBtn').disabled = false;
    input.focus();
}

// Init
newChat();
renderHistory();
</script>
</body>
</html>"""
    return HTMLResponse(content=html)

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
    session_turn = len(session_store.get(request.session_id, []))
    if len(request.question.strip()) < 3 and session_turn == 0:
        raise HTTPException(
            status_code=400,
            detail="Please ask a complete question."
        )

    start_time = time.time()

    result = process_query(
        user_question=request.question,
        session_id=request.session_id,
        n_results=request.n_results,
        department_filter=request.department_filter
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

    # Generate follow-up questions only for successful non-guardrail responses
    follow_ups = []
    if not result["guardrail_triggered"]:
        follow_ups = generate_follow_up_questions(
            result["question"],
            result["answer"]
        )

    return {
        "question": result["question"],
        "answer": result["answer"],
        "sources": result["sources"],
        "conversation_turn": result["conversation_turn"],
        "guardrail_triggered": result["guardrail_triggered"],
        "guardrail_reason": result["guardrail_reason"],
        "rerank_method": result["rerank_method"],
        "response_time_seconds": response_time,
        "follow_up_questions": follow_ups
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