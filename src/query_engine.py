import os
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()

# Paths - same as ingest.py so ChromaDB location matches exactly
BASE_DIR = Path(__file__).parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"

# Connect to the SAME ChromaDB we created in ingest.py
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="clinical_docs",
    metadata={"hnsw:space": "cosine"}
)

# Load the SAME embedding model used in ingest.py
# CRITICAL: Must be same model — different model = wrong results
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ─────────────────────────────────────────────
# Reranker — Graceful Fallback
# Tries CrossEncoder first (best quality)
# Falls back to bi-encoder if unavailable
# CrossEncoder crashes or produces nan on Apple Silicon M1/M2/M3
# Bi-encoder fallback works on all platforms
# On Intel/Linux/Windows CrossEncoder works fully
# ─────────────────────────────────────────────
print("Loading reranker model...")
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    USE_CROSS_ENCODER = True
    print("CrossEncoder reranker loaded successfully.")
except Exception as e:
    print(f"CrossEncoder unavailable ({e}). Using bi-encoder fallback.")
    USE_CROSS_ENCODER = False

# Connect to Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.2
)

# Conversation memory
# Stores full session history in memory
# Resets when script stops
conversation_history = []


def check_guardrails(question: str, relevance_scores: list) -> dict:
    """
    Checks question and relevance scores against safety rules.
    Runs BEFORE sending anything to LLM.
    Returns:
        - is_safe: True if question can proceed to LLM
        - reason: explanation if blocked
        - response: pre-built response if blocked
    """
    question_lower = question.lower().strip()

    # Rule 1 — Block direct medical advice requests
    medical_advice_triggers = [
        "should i",
        "should my patient",
        "can i give",
        "is it safe to",
        "can my patient",
        "what dose should",
        "recommend for my",
        "advise me on",
        "tell me if i should",
        "is this safe for"
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

    # Rule 2 — Block completely off-topic questions
    off_topic_triggers = [
        "stock price",
        "share price",
        "weather",
        "football",
        "recipe",
        "movie",
        "politics",
        "election",
        "celebrity",
        "social media",
        "cryptocurrency",
        "bitcoin"
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

    # Rule 3 — Block when all relevance scores are too low
    if relevance_scores and all(score < 0.35 for score in relevance_scores):
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

    # All rules passed
    return {
        "is_safe": True,
        "reason": "passed",
        "response": None
    }


def get_biencoder_scores(question: str, chunks: list) -> list:
    """
    Bi-encoder reranking fallback.
    Uses same embedder as ingest.py.
    Works on all platforms including Apple Silicon.
    """
    question_vec = embedder.encode(
        question,
        convert_to_tensor=True
    )
    chunk_vecs = embedder.encode(
        chunks,
        convert_to_tensor=True
    )
    return [
        float(util.cos_sim(question_vec, chunk_vec)[0][0])
        for chunk_vec in chunk_vecs
    ]


def query_rag(user_question: str, n_results: int = 5):
    """
    Main RAG function with reranking, guardrails and conversation memory:
    1. Embed the user question
    2. Search ChromaDB for top 20 candidates
    3. Rerank top 20 — CrossEncoder or bi-encoder fallback
    4. Check guardrails
    5. Build augmented prompt with conversation history
    6. Send to Groq LLM
    7. Store question and answer in history
    8. Return answer with sources
    """

    # Step 1 — Convert user question to vector
    question_embedding = embedder.encode(user_question).tolist()

    # Step 2 — Search ChromaDB for top 20 candidates
    # Fetch 20 to give reranker more candidates to work with
    raw_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=20,
        include=["documents", "metadatas", "distances"]
    )

    # Step 3 — Extract raw candidates
    raw_chunks = raw_results["documents"][0]
    raw_metadatas = raw_results["metadatas"][0]
    raw_distances = raw_results["distances"][0]

    # Step 4 — Reranking with graceful fallback
    # CrossEncoder: reads question+chunk together, most accurate
    # Bi-encoder: uses embedder, works on all platforms
    if USE_CROSS_ENCODER:
        try:
            rerank_pairs = [[user_question, chunk] for chunk in raw_chunks]
            raw_scores = [float(s) for s in reranker.predict(rerank_pairs)]

            # Detect nan scores — happens on Apple Silicon M1/M2/M3
            # nan never equals itself in Python — this is the correct check
            if any(s != s for s in raw_scores):
                raise ValueError("CrossEncoder produced nan scores on this platform")

            rerank_scores = raw_scores
            rerank_method = "cross-encoder"

        except Exception as e:
            print(f"\nCrossEncoder inference failed ({e}). Using bi-encoder.")
            rerank_scores = get_biencoder_scores(user_question, raw_chunks)
            rerank_method = "bi-encoder-fallback"
    else:
        rerank_scores = get_biencoder_scores(user_question, raw_chunks)
        rerank_method = "bi-encoder"

    # Step 5 — Combine and sort by rerank score highest first
    ranked_results = sorted(
        zip(rerank_scores, raw_chunks, raw_metadatas, raw_distances),
        key=lambda x: x[0],
        reverse=True
    )

    # Take top 5 after reranking
    top_results = ranked_results[:n_results]

    # Step 6 — Build context from reranked top 5
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

    # Step 7 — Guardrails check
    # Runs BEFORE sending to LLM
    guardrail_result = check_guardrails(user_question, relevance_scores)

    if not guardrail_result["is_safe"]:
        print(f"\n⚠️  Guardrail triggered: {guardrail_result['reason']}")
        return {
            "question": user_question,
            "answer": guardrail_result["response"],
            "sources": [],
            "conversation_turn": len(conversation_history) + 1,
            "guardrail_triggered": True,
            "guardrail_reason": guardrail_result["reason"]
        }

    # Step 8 — System prompt
    system_prompt = """You are a clinical document assistant at Siemens Healthineers.
Your job is to answer questions based strictly on the provided document context.
Always be precise and professional.
If the answer is not found in the context, respond with:
'I could not find relevant information in the available documents.'
Never make up information that is not in the context."""

    # Step 9 — User prompt with context
    user_prompt = f"""CONTEXT FROM INTERNAL DOCUMENTS:
{context}

USER QUESTION:
{user_question}

Please provide a clear, professional answer based only on the context above.
At the end, mention which documents you used."""

    # Step 10 — Build messages with conversation history
    messages = [SystemMessage(content=system_prompt)]

    for turn in conversation_history:
        messages.append(HumanMessage(content=turn["question"]))
        messages.append(SystemMessage(content=turn["answer"]))

    messages.append(HumanMessage(content=user_prompt))

    # Step 11 — Send to Groq LLM
    response = llm.invoke(messages)

    # Step 12 — Save to conversation history
    conversation_history.append({
        "question": user_question,
        "answer": response.content
    })

    # Step 13 — Return everything
    return {
        "question": user_question,
        "answer": response.content,
        "sources": sources,
        "conversation_turn": len(conversation_history),
        "guardrail_triggered": False,
        "guardrail_reason": "passed",
        "rerank_method": rerank_method
    }


# Interactive loop
if __name__ == "__main__":
    print("\n" + "="*60)
    print("ClinicalRAG Query Engine — Interactive Mode")
    print("="*60)
    print("Commands:")
    print("  'exit'  → quit the program")
    print("  'clear' → reset conversation memory")
    print("="*60 + "\n")

    while True:
        user_input = input("Your question: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Shutting down ClinicalRAG. Goodbye.")
            break

        if user_input.lower() == "clear":
            conversation_history.clear()
            print("Conversation memory cleared.\n")
            continue

        print("\nSearching documents and generating answer...\n")

        result = query_rag(user_input)

        print(f"Turn {result['conversation_turn']} | ANSWER:")
        print("-"*60)
        print(result["answer"])

        if result["sources"]:
            print("\nSOURCES USED:")
            print("-"*60)
            for source in result["sources"]:
                print(
                    f"Chunk {source['chunk_number']}: "
                    f"{source['source']} "
                    f"(Page {source['page']}) | "
                    f"Dept: {source['department']} | "
                    f"Cosine: {source['cosine_relevance']} | "
                    f"Rerank: {source['rerank_score']}"
                )
            print(f"\nRerank method used: {result['rerank_method']}")
        print()