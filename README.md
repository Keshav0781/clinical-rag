# ClinicalRAG — Intelligent Document Search System

> Production RAG pipeline for Siemens Healthineers clinical document search, deployed on GCP Cloud Run.

[![Live API](https://img.shields.io/badge/Live%20API-GCP%20Cloud%20Run-blue)](https://clinical-rag-service-324111066236.europe-west1.run.app/docs)
[![Python](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-28.0-2496ED)](https://docker.com)

---

## What It Does

ClinicalRAG allows medical professionals to search and query Siemens Healthineers clinical documents using natural language. Instead of manually reading through technical PDFs, users ask questions and receive precise answers with source references in under 3 seconds.

**Live Demo:** https://clinical-rag-service-324111066236.europe-west1.run.app/docs

---

## Architecture

```
User Question
      ↓
FastAPI REST API (GCP Cloud Run)
      ↓
Embedding Model (all-MiniLM-L6-v2)
      ↓
ChromaDB Vector Search → Top 20 candidates
      ↓
CrossEncoder Reranker → Top 5 chunks
      ↓
Safety Guardrails (3 rules)
      ↓
Groq LLM (Llama 3.3 70B) + Conversation Memory
      ↓
Answer + Sources + Metadata
      ↓
Async Audit Log → BigQuery
```

---

## Key Features

- **Two-stage retrieval** — semantic search retrieves 20 candidates, CrossEncoder reranker selects best 5
- **Conversation memory** — session-based multi-turn conversations with follow-up question support
- **Safety guardrails** — blocks medical advice requests, off-topic questions, and low-relevance queries
- **Weak retrieval detection** — automatically switches to memory-only prompt when retrieval scores are negative
- **Async audit logging** — every query logged to BigQuery with retry logic and exponential backoff, never blocking API response
- **Production Docker** — multi-stage build, non-root user, security vulnerability scanning, pinned base image
- **GCP deployment** — Cloud Run with min-instances, service account least privilege, Artifact Registry

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.11 |
| API Framework | FastAPI + Uvicorn | 0.136 / 0.46 |
| Vector Database | ChromaDB | 1.5.9 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | 5.4.1 |
| Reranker | CrossEncoder (ms-marco-MiniLM-L-6-v2) | 5.4.1 |
| LLM | Groq API — Llama 3.3 70B | — |
| LLM Framework | LangChain + LangChain-Groq | 1.2.17 |
| Audit Logging | Google BigQuery | 3.27.0 |
| Retry Logic | Tenacity | 8.2.3 |
| Container | Docker (multi-stage, linux/amd64) | 28.0 |
| Cloud Hosting | GCP Cloud Run | europe-west1 |
| Image Registry | GCP Artifact Registry | europe-west1 |

---

## Project Structure

```
clinical-rag/
├── src/
│   ├── ingest.py          # Phase 1 — PDF ingestion, chunking, embedding, ChromaDB storage
│   ├── query_engine.py    # Phase 2 — RAG pipeline, reranking, guardrails, conversation memory
│   ├── api.py             # Phase 3 — FastAPI REST API with session management
│   └── logger.py          # Phase 4 — Async BigQuery audit logging with retry logic
├── data/
│   └── chroma/            # ChromaDB persistent storage (excluded from Git)
├── documents/             # Source PDFs (excluded from Git — copyrighted)
├── logs/                  # Audit log exports (excluded from Git)
├── Dockerfile             # Multi-stage production build
├── .dockerignore          # Excludes venv, .env, chroma data from Docker context
├── Makefile               # Shortcuts: make build, make run-local, make push, make deploy
├── requirements.txt       # 17 direct dependencies
└── README.md              # This file
```

---

## API Endpoints

### GET /health
Returns system status and configuration.

```json
{
  "status": "healthy",
  "total_chunks": 834,
  "rerank_method": "cross-encoder",
  "groq_model": "llama-3.3-70b-versatile",
  "api_version": "1.0.0"
}
```

### POST /query
Accepts a question and session_id, returns answer with sources.

**Request:**
```json
{
  "question": "What is xSPECT Bone and what are its clinical benefits?",
  "session_id": "your-session-id"
}
```

**Response:**
```json
{
  "question": "What is xSPECT Bone and what are its clinical benefits?",
  "answer": "xSPECT Bone is a technology that...",
  "sources": [...],
  "conversation_turn": 1,
  "guardrail_triggered": false,
  "guardrail_reason": "passed",
  "rerank_method": "cross-encoder",
  "response_time_seconds": 2.2
}
```

### POST /ingest
Triggers re-ingestion of all PDFs in the documents/ folder.

### DELETE /session/{session_id}
Clears conversation history for a specific session.

---

## How to Run Locally

### Prerequisites
- Python 3.11
- Docker 28+
- Groq API key (free at console.groq.com)

### 1. Clone the repository
```bash
git clone https://github.com/Keshav0781/clinical-rag.git
cd clinical-rag
```

### 2. Set up environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 4. Add your documents
```bash
# Place your PDF files in the documents/ folder
# Then run ingestion
python src/ingest.py
```

### 5. Start the API
```bash
uvicorn src.api:app --reload --port 8000
```

### 6. Test at Swagger UI
Open http://localhost:8000/docs

---

## Running with Docker

```bash
# Build
make build

# Run locally
docker run -p 8080:8080 --env-file .env clinical-rag:v8

# Test
curl http://localhost:8080/health
```

---

## Try the Live API

Open the Swagger UI and test these three queries in order:

**Query 1 — Document search:**
```json
{
  "question": "What is xSPECT Bone and what are its clinical benefits?",
  "session_id": "demo-001"
}
```

**Query 2 — Conversation memory (same session_id):**
```json
{
  "question": "Can you elaborate on the first benefit you mentioned?",
  "session_id": "demo-001"
}
```

**Query 3 — Safety guardrail:**
```json
{
  "question": "Should I take ibuprofen for my headache?",
  "session_id": "demo-001"
}
```

---

## Known Improvements / Next Steps

- [ ] Replace in-memory session store with Redis for persistence across container restarts
- [ ] Add CI/CD pipeline — GitHub Actions for automated build, scan, and deploy
- [ ] Move GROQ_API_KEY to GCP Secret Manager
- [ ] Implement Terraform for infrastructure as code
- [ ] Add Power BI dashboard connected to BigQuery audit logs
- [ ] Add automated pytest test suite
- [ ] Implement dead letter queue for failed BigQuery writes using Cloud Pub/Sub

---

## About

Built as a portfolio project demonstrating production AI engineering skills:
- RAG pipeline design and implementation
- Vector database setup and management
- REST API development and deployment
- Docker containerization with security best practices
- GCP cloud deployment and monitoring
- Async audit logging and analytics pipeline

**Author:** Keshav Jha — M.Sc. Data Science, FAU Erlangen-Nuremberg
**Target Role:** AI Engineer — Germany

---

## License

MIT License — see [LICENSE](LICENSE) for details.
