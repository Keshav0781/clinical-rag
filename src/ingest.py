import os
import sys
from pathlib import Path
from datetime import date
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent.parent
DOCS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "data" / "chroma"

# Initialize ChromaDB - same collection name as before
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="clinical_docs",
    metadata={"hnsw:space": "cosine"}
)

# Initialize embedding model - same as before
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Text splitter - same settings as before
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# ─────────────────────────────────────────────
# NEW ADDITION 1 — Quality Validation Function
# ─────────────────────────────────────────────
def is_valid_chunk(text: str) -> bool:
    """
    Validates chunk quality before storing in ChromaDB.
    Filters out noise like page numbers, headers, copyright lines.
    Returns True if chunk is valid, False if it should be skipped.
    """
    # Remove whitespace for accurate length check
    cleaned = text.strip()

    # Rule 1: Skip if too short (less than 50 characters)
    if len(cleaned) < 50:
        return False

    # Rule 2: Skip if too few words (less than 10 words)
    word_count = len(cleaned.split())
    if word_count < 10:
        return False

    # Rule 3: Skip common noise patterns
    noise_patterns = [
        "table of contents",
        "all rights reserved",
        "©",
        "siemens healthineers",
        "www.",
        "http",
        "page ",
        "figure ",
        "references"
    ]
    lower_text = cleaned.lower()
    # Skip if chunk is ONLY a noise pattern (short noise lines)
    if any(lower_text.startswith(pattern) for pattern in noise_patterns):
        if len(cleaned) < 100:
            return False

    return True


# ─────────────────────────────────────────────
# NEW ADDITION 2 — Department Detection Function
# ─────────────────────────────────────────────
def detect_department(filename: str) -> str:
    """
    Detects department from PDF filename.
    Returns department name for metadata.
    """
    filename_lower = filename.lower()

    if any(term in filename_lower for term in ["spect", "xspect", "iqspect", "mi_", "_mi"]):
        return "Molecular Imaging"
    elif any(term in filename_lower for term in ["oncology", "cancer", "tumor"]):
        return "Oncology"
    elif any(term in filename_lower for term in ["operational", "excellence", "operations"]):
        return "Operations"
    elif any(term in filename_lower for term in ["rad", "radiology", "imaging"]):
        return "Radiology Services"
    else:
        return "General Clinical"


def ingest_documents():
    pdf_files = list(DOCS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in documents/ folder")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF files")

    # Track statistics
    total_chunks_attempted = 0
    total_chunks_stored = 0
    total_chunks_skipped = 0

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        print(f"\nProcessing: {pdf_path.name}")

        # Load PDF
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        # Split into chunks
        chunks = text_splitter.split_documents(pages)
        print(f"  → {len(chunks)} chunks created")

        # NEW: Detect department from filename
        department = detect_department(pdf_path.name)

        # Generate embeddings and store in ChromaDB
        for i, chunk in enumerate(chunks):
            total_chunks_attempted += 1

            # NEW ADDITION 1: Validate chunk quality
            if not is_valid_chunk(chunk.page_content):
                total_chunks_skipped += 1
                continue

            chunk_id = f"{pdf_path.stem}_{i}"
            embedding = embedder.encode(chunk.page_content).tolist()

            # NEW ADDITION 2: Enhanced metadata
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk.page_content],
                metadatas=[{
                    # Original metadata
                    "source": pdf_path.name,
                    "page": chunk.metadata.get("page", 0),
                    # New enhanced metadata
                    "document_type": "Clinical White Paper",
                    "department": department,
                    "date_processed": str(date.today()),
                    "word_count": len(chunk.page_content.split())
                }]
            )
            total_chunks_stored += 1

    print(f"\n✅ Ingestion complete!")
    print(f"  Total chunks attempted : {total_chunks_attempted}")
    print(f"  Total chunks stored    : {total_chunks_stored}")
    print(f"  Total chunks skipped   : {total_chunks_skipped} (noise filtered)")
    print(f"  Final ChromaDB count   : {collection.count()}")


if __name__ == "__main__":
    ingest_documents()